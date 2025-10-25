from forge.actors.generator import Generator as Policy
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.reference_model import ReferenceModel
from forge.actors.trainer import RLTrainer
from apps.grpo.main import DatasetActor, RewardActor, ComputeAdvantages
from forge.data.rewards import MathReward, ThinkingReward
import asyncio
import torch

model = "Qwen/Qwen2.5-0.5B-Instruct"
group_size = 1

(
    dataloader,
    policy,
    trainer,
    replay_buffer,
    compute_advantages,
    ref_model,
    reward_actor,
) = await asyncio.gather(
        # Dataset actor (CPU)
        DatasetActor.options(procs=1).as_actor(
            path="openai/gsm8k",
            revision="main",
            data_split="train",
            streaming=True,
            model=model,
        ),
        # Policy service with GPU
        Policy.options(procs=1, with_gpus=True, num_replicas=1).as_service(
            engine_config={
                "model": model,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "enforce_eager": False
            },
            sampling_config={
                "n": group_size,
                "max_tokens": 16,
                "temperature": 1.0,
                "top_p": 1.0
            }
        ),
        # Trainer actor with GPU
        RLTrainer.options(procs=1, with_gpus=True).as_actor(
            # Trainer config would come from YAML in real usage
            model={"name": "qwen3", "flavor": "1.7B", "hf_assets_path": f"hf://{model}"},
            optimizer={"name": "AdamW", "lr": 1e-5},
            training={"local_batch_size": 2, "seq_len": 2048}
        ),
        # Replay buffer (CPU)
        ReplayBuffer.options(procs=1).as_actor(
            batch_size=2,
            max_policy_age=1,
            dp_size=1
        ),
        # Advantage computation (CPU)
        ComputeAdvantages.options(procs=1).as_actor(),
        # Reference model with GPU
        ReferenceModel.options(procs=1, with_gpus=True).as_actor(
            model={"name": "qwen3", "flavor": "1.7B", "hf_assets_path": f"hf://{model}"},
            training={"dtype": "bfloat16"}
        ),
        # Reward actor (CPU)
        RewardActor.options(procs=1, num_replicas=1).as_service(
            reward_functions=[MathReward(), ThinkingReward()]
        )
    )