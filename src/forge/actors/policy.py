# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import os
import sys
import time
from collections.abc import Mapping
from copy import copy
from dataclasses import asdict, dataclass, field, fields

import torch
import torchstore as ts
from monarch.actor import current_rank, endpoint, ProcMesh
from torchstore.state_dict_utils import DELIM
from vllm.config import VllmConfig

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.utils import _validate_truncation_size
from vllm.executor.multiproc_worker_utils import set_multiprocessing_worker_envs
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import GuidedDecodingParams, RequestOutputKind, SamplingParams
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import get_distributed_init_method
from vllm.v1.core.kv_cache_utils import get_kv_cache_config
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.engine.processor import Processor
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.worker.worker_base import WorkerWrapperBase

from forge.controller import ForgeActor, get_proc_mesh, stop_proc_mesh

from forge.data.sharding import VLLMSharding
from forge.interfaces import Policy as PolicyInterface
from forge.types import ProcessConfig


@dataclass
class SamplingConfig:
    """
    Overrides for vLLMs sampling params.

    Note: We'll want to tie this closer to or directly use vllm's
            SamplingParams. It is currently used to track a supported
            subset

    Args:
        n: Number of samples to generate.
        guided_decoding: Whether to use guided decoding.
        max_tokens: Maximum number of tokens to generate.
    """

    n: int = 1
    guided_decoding: bool = False
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0

    def __post_init__(self):
        gd_params = None
        if self.guided_decoding:
            gd_params = GuidedDecodingParams(choice=["Positive", "Negative"])
        self.guided_decoding = gd_params

    @classmethod
    def from_dict(cls, d: Mapping):
        d = dict(d)
        all_fields = set(cls.__dataclass_fields__.keys())
        valid_args = {k: v for k, v in d.items() if k in all_fields}
        return cls(**valid_args)


@dataclass
class EngineConfig(EngineArgs):
    """
    EngineConfig extends EngineArgs with worker-specific fields.
    Overlapping keys in input dict will override EngineArgs defaults.
    """

    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enforce_eager: bool = False

    # Original method returns False when not run in the main thread
    _is_v1_supported_oracle = lambda *_: True

    @classmethod
    def from_dict(cls, d: Mapping):
        d = dict(d)
        all_fields = [f.name for f in fields(cls)]
        valid_args = {k: v for k, v in d.items() if k in all_fields}
        return cls(**valid_args)

    def create_vllm_config(self) -> VllmConfig:
        # This is not a typo: EngineArgs.create_engine_config
        # creates a VllmConfig
        return self.create_engine_config(UsageContext.LLM_CLASS)


@dataclass
class Policy(PolicyInterface):
    engine_config: EngineConfig | Mapping = field(default_factory=EngineConfig)
    sampling_config: SamplingConfig | Mapping = field(default_factory=SamplingConfig)
    available_devices: str | None = None
    # Gets set up by setup
    sampling_params: SamplingParams | None = None
    lora_request: LoRARequest | None = None
    tokenization_kwargs: dict = field(default_factory=dict)
    policy_worker: "PolicyWorker" = None

    def __post_init__(self):
        self._run_task: asyncio.Task | None = None
        self._policy_proc: ProcMesh | None = None
        self._worker_procs: ProcMesh | None = None
        self.weights_version: int = 0
        self.running = False
        if isinstance(self.engine_config, Mapping):
            self.engine_config = EngineConfig.from_dict(self.engine_config)
        if isinstance(self.sampling_config, Mapping):
            self.sampling_config = SamplingConfig.from_dict(self.sampling_config)

    @classmethod
    async def launch(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls: type["Policy"],
        *,
        process_config: ProcessConfig,
        engine_config: EngineConfig | Mapping = EngineConfig(),
        sampling_config: SamplingConfig | Mapping = SamplingConfig(),
        available_devices: str | None = None,
        **kwargs,
    ) -> "Policy":
        # Note - get_proc_mesh will set MASTER_ADDR, MASTER_PORT and CUDA_VISIBLE_DEVICES
        # automatically.
        worker_procs = await get_proc_mesh(process_config=process_config)

        # TODO - we will want to ensure colocation with workers
        policy_proc_config = copy(process_config)
        policy_proc_config.num_procs = 1
        policy_proc_config.with_gpus = False

        policy_proc = await get_proc_mesh(process_config=policy_proc_config)

        if isinstance(engine_config, Mapping):
            engine_config = EngineConfig.from_dict(engine_config)

        vllm_config = engine_config.create_vllm_config()
        workers = await worker_procs.spawn(
            "vllm_worker", PolicyWorker, vllm_config=vllm_config
        )

        if isinstance(sampling_config, Mapping):
            sampling_config = SamplingConfig(**sampling_config)

        # TODO - expand support so name can stick within kwargs
        actor_name = kwargs.pop("name", cls.__name__)
        policy = await policy_proc.spawn(
            actor_name,
            cls,
            engine_config=engine_config,
            sampling_config=sampling_config,
            available_devices=available_devices,
            policy_worker=workers,
        )
        policy._policy_proc = policy_proc
        policy._worker_procs = worker_procs
        await policy.setup.call()
        return policy

    @classmethod
    async def shutdown(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls: type["Policy"], actor: "Policy"
    ):
        assert (
            actor._policy_proc is not None
        ), "Tried to shutdown a policy that was not initialized correctly"
        assert (
            actor._worker_procs is not None
        ), "Tried to shutdown a policy that was not initialized correctly"

        # TODO - may want to expand stop to gracefully respond to
        # ongoing requests.
        await actor.stop.call()
        await stop_proc_mesh(actor._worker_procs)
        await stop_proc_mesh(actor._policy_proc)

    @endpoint
    async def setup(self):
        # Set up policy_worker
        assert self.policy_worker is not None, "Policy worker should not be None"
        await self.policy_worker.setup.call()

        self.request_id = 0
        self.requests: Dict[str, tuple[None | ParentRequest, asyncio.Future]] = {}
        self.vllm_config: VllmConfig = self.engine_config.create_vllm_config()

        # Setup sampling params
        self.sampling_params = get_default_sampling_params(
            self.vllm_config, overrides=asdict(self.sampling_config)
        )

        # Setup processors
        # TODO: move all processing to the Environment
        # TODO: add support for `log_stats` and `mm_registry`
        tokenizer = init_tokenizer_from_configs(
            model_config=self.vllm_config.model_config,
            scheduler_config=self.vllm_config.scheduler_config,
            lora_config=self.vllm_config.lora_config,
        )
        self.processor = Processor(
            vllm_config=self.vllm_config, tokenizer=tokenizer, mm_registry=None
        )
        self.output_processor = OutputProcessor(tokenizer, log_stats=None)

        # Setup scheduler
        # TODO: Add support for `log_stats`
        kv_cache_configs = await self.policy_worker.setup_kv_cache.call()
        kv_cache_config = kv_cache_configs._values[0]
        self.vllm_config.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.vllm_config.cache_config.num_cpu_blocks = 0

        structured_output_manager = StructuredOutputManager(self.vllm_config)
        self.scheduler = Scheduler(
            vllm_config=self.vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            include_finished_set=False,
            log_stats=None,
        )
        self.start_processing()

    def start_processing(self):
        """Start the replica's processing loop if not already running."""
        if self._run_task is None or self._run_task.done():
            self._run_task = asyncio.create_task(self.run())

    @endpoint
    async def generate(self, prompt: str, priority: int = 0) -> RequestOutput:
        """Generate a response for the given prompt

        Args:
            prompt (str): The prompt to generate a response for.
            priority (int, optional): The priority of the request. Defaults to 0.

        Returns:
            RequestOutput: vLLM class with the generated response.
        """
        self.request_id += 1 % sys.maxsize
        request_id = str(self.request_id)  # implement from a counter

        # Wraps prompt into a dict
        prompt_dict: dict[str, str] = convert_input(prompt=prompt)

        # truncate prmpt
        tokenization_kwargs = self.tokenization_kwargs or {}
        # TODO: add truncation support https://github.com/vllm-project/vllm/issues/4507
        truncate_prompt_tokens = self.sampling_params.truncate_prompt_tokens
        _validate_truncation_size(
            self.vllm_config.model_config.max_model_len,
            truncate_prompt_tokens,
            tokenization_kwargs,
        )

        # process and tokenize prompt
        prompt_str, request = self.processor.process_inputs(
            request_id=request_id,
            prompt=prompt_dict,
            params=self.sampling_params,
            arrival_time=None,
            lora_request=self.lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=None,
            priority=priority,
            data_parallel_rank=None,
        )

        # Explicitly keeping the redundant logic to make it easier to pick up
        # vllm changes
        # TODO: Clean up before release
        if (num_samples := self.sampling_params.n) == 1:
            self.output_processor.add_request(request, prompt_str, None, 0)
            request, _ = self.preprocess_add_request(request)
            request_fut = asyncio.Future()
            self.requests[request_id] = (None, request_fut)

            self.scheduler.add_request(request)
        else:
            parent_req = ParentRequest(request_id, self.sampling_params)
            for idx in range(num_samples):
                # Note: `get_child_info` mutates ParentRequest to track the
                # generated child request
                child_request_id, params = parent_req.get_child_info(idx)
                child_request = request if idx == num_samples - 1 else copy(request)
                child_request.request_id = child_request_id
                child_request.sampling_params = params
                self.output_processor.add_request(
                    child_request, prompt_str, parent_req, idx
                )
                child_request, _ = self.preprocess_add_request(child_request)

                self.scheduler.add_request(child_request)
            request_fut = asyncio.Future()
            self.requests[request_id] = (parent_req, request_fut)

        return await request_fut

    # Abstracted to match vllm
    # https://github.com/vllm-project/vllm/blob/0e3bb543f064eb416bca4f6f3013efa3830b12f7/vllm/v1/engine/core.py#L419
    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        if request.mm_hashes is not None:
            raise NotImplementedError("Support for mm_hash is not implemented yet.")
        request: Request = Request.from_engine_core_request(request)
        if request.use_structured_output:
            self.scheduler.structured_output_manager.grammar_init(request)

        return request, 0  # Unused Arg: Current Wave

    async def run(self):
        # TODO: add support for `iteration_stats`
        # TODO: move postprocessing out of loop to not block
        parallel_config = self.vllm_config.parallel_config
        output_rank = parallel_config.world_size - parallel_config.tensor_parallel_size
        self.running = True
        while self.running:
            scheduler_output = self.scheduler.schedule()
            worker_outputs = await self.policy_worker.execute_model.call(
                scheduler_output
            )
            worker_output = worker_outputs._values[output_rank]
            outputs = self.scheduler.update_from_output(scheduler_output, worker_output)
            outputs = outputs.get(0) or EngineCoreOutputs()
            await asyncio.sleep(0)  # Release control before processing outputs

            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                engine_core_timestamp=outputs.timestamp,
                iteration_stats=None,
            )

            for request_output in processed_outputs.request_outputs:
                if request_output.finished:
                    _, fut = self.requests.pop(request_output.request_id)
                    fut.set_result(request_output)

    @endpoint
    async def update_weights(self):
        # TODO: If generating long sequences, this might be long and will block policy weight updates
        curr_requests = [fut for _, fut in self.requests.values()]
        if curr_requests:
            self.logger.debug(f"Waiting for {len(curr_requests)} pending requests")
            await asyncio.gather(*curr_requests)

        self.logger.debug(f"Starting weight update on {self.__class__.__name__}")
        await self.policy_worker.update.call(version=self.weights_version)
        self.weights_version += 1
        self.logger.info(f"Weight update completed (now v{self.weights_version})")

    @endpoint
    async def _get_model_params(self) -> dict[str, torch.Tensor]:
        """Get the current model parameters. Only for testing purposes."""
        model_params = await self.policy_worker._get_model_params.choose()
        return model_params

    @endpoint
    async def get_version(self) -> int:
        """Get the current policy version."""
        return self.weights_version

    @endpoint
    async def stop(self):
        self.running = False


@dataclass
class PolicyWorker(ForgeActor):
    vllm_config: VllmConfig
    state_dict_key: str = "model_state_dict"

    @endpoint
    async def setup(self):
        # TODO: remove ["gpus"] when monarch implements a flat rank
        self.rank = current_rank()["gpus"]
        self.worker = self.setup_worker()

    @endpoint
    async def execute_model(self, schedule: SchedulerOutput):
        return self.worker.execute_model(schedule)

    async def _load_tensor_parallel_state_dict(
        self, current_state_dict: dict, version: int
    ):
        """
        Load full state dict from torchstore into tensor parallel model with deterministic sharding.
        """
        sharding = VLLMSharding(
            self.vllm_config.parallel_config.tensor_parallel_size, self.rank
        )

        for param_name in current_state_dict.keys():
            current_tensor = current_state_dict[param_name]

            # Load the full tensor from torchstore
            # TODO: only get the part of the tensor that is needed
            stored_tensor = await ts.get(
                f"{self.state_dict_key}{DELIM}{version}{DELIM}{param_name}"
            )
            sharding.load_from_source_to_target(
                param_name,
                stored_tensor,
                current_tensor,
            )

    @endpoint
    async def update(self, version: int):
        """Update model weights by reading state dict from torchstore"""
        key = f"{self.state_dict_key}{DELIM}{version}"
        model = self.worker.model_runner.model
        current_state_dict = model.state_dict()
        start = time.time()
        await self._load_tensor_parallel_state_dict(current_state_dict, version)
        self.logger.debug(
            f"Loaded state dict from {key} in {time.time() - start} seconds"
        )

    @endpoint
    async def setup_kv_cache(self):
        """Based on vllm/v1/engine/core.py:EngineCore._initialize_kv_caches
        TODO: test that fails if vllm method updates
        """
        kv_cache_spec = self.worker.get_kv_cache_spec()
        if kv_cache_spec is not None:
            available_gpu_memory = self.worker.determine_available_memory()
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = 0

        # Get the kv cache tensor size
        kv_cache_config = get_kv_cache_config(
            self.vllm_config, kv_cache_spec, available_gpu_memory
        )
        # TODO: unify configs across TorchStore
        # unify_kv_cache_configs(kv_cache_configs)
        self.vllm_config.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.vllm_config.cache_config.num_cpu_blocks = 0

        # Initialize kv cache and warmup the execution:
        # from multiproc_executor.py:MultiprocExecutor.initialize_from_config
        kv_cache_configs = [None] * self.vllm_config.parallel_config.world_size
        kv_cache_configs[self.rank] = kv_cache_config
        self.worker.initialize_from_config(kv_cache_configs)
        self.worker.compile_or_warm_up_model()
        self.worker.initialize_cache(kv_cache_config.num_blocks, 0)
        return kv_cache_config

    @endpoint
    async def get_vllm_config(self) -> VllmConfig:
        return self.vllm_config

    @endpoint
    async def _get_model_params(self) -> dict[str, torch.Tensor]:
        model = self.worker.model_runner.model
        state_dict = {}

        for name, param in model.named_parameters():
            if "layers.0" not in name:
                continue
            state_dict[name] = param.cpu().detach()
        return state_dict

    def setup_worker(self):
        """Build and Instantiate vLLM worker"""
        parallel_config = self.vllm_config.parallel_config
        set_multiprocessing_worker_envs(parallel_config)
        ip, port = os.getenv("MASTER_ADDR"), os.getenv("MASTER_PORT")
        distributed_init_method = get_distributed_init_method(ip, port)
        all_kwargs = [{}] * parallel_config.world_size
        local_rank = self.rank % torch.accelerator.device_count()
        is_driver_worker = self.rank % parallel_config.tensor_parallel_size == 0
        all_kwargs[self.rank] = {
            "vllm_config": self.vllm_config,
            "local_rank": local_rank,
            "rank": self.rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
        }
        worker = WorkerWrapperBase(self.vllm_config, self.rank)
        worker.init_worker(all_kwargs)
        worker.init_device()
        worker.load_model()
        return worker


def convert_input(prompt=None, prompt_token_ids=None) -> dict:
    assert (prompt is None) ^ (prompt_token_ids is None)
    if prompt is not None:
        return {"prompt": prompt}
    return {"prompt_token_ids": prompt_token_ids}


def get_default_sampling_params(vllm_config, overrides=None) -> SamplingParams:
    default_params = vllm_config.model_config.get_diff_sampling_param()
    if overrides is not None:
        default_params |= overrides
    if default_params:
        params = SamplingParams.from_optional(**default_params)
    else:
        params = SamplingParams()
    # We only care about the final output
    params.output_kind = RequestOutputKind.FINAL_ONLY
    return params
