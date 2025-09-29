# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import logging
import time

from forge.controller.actor import ForgeActor
from forge.controller.provisioner import shutdown
from forge.observability.metric_actors import setup_metric_logger
from forge.observability.metrics import record_metric, ReductionType

from monarch.actor import current_rank, endpoint

logging.basicConfig(level=logging.DEBUG)


class TrainActor(ForgeActor):
    """Example training actor that records loss metrics."""

    @endpoint
    async def train_step(self, step: int):
        rank = current_rank().rank
        value = rank * 1000 + 100 * step
        print(f"[TRAIN] Rank {rank}: Step {step}, loss={value}")
        record_metric("train/loss", value)


class GeneratorActor(ForgeActor):
    """Example generation actor that records token count metrics."""

    @endpoint
    async def generate_step(self, step: int, substep: int):
        rank = current_rank().rank
        value = rank * 1000 + step * 100 + substep * 10
        print(f"[GEN] Rank {rank}: Step {step}.{substep}, tokens={value}")
        record_metric("generate/tokens", value, ReductionType.SUM)


# Main
async def main():
    """Example demonstrating distributed metric logging with different backends."""
    group = f"grpo_exp_{int(time.time())}"

    # Config format: {backend_name: backend_config_dict}
    # Each backend can specify reduce_across_ranks to control distributed logging behavior
    config = {
        "console": {"reduce_across_ranks": True},
        "wandb": {
            "project": "my_project",
            "group": group,
            "reduce_across_ranks": True,
            # Only useful if NOT reduce_across_ranks.
            "share_run_id": False,  # Share run ID across ranks -- Not recommended.
        },
    }

    service_config = {"procs": 2, "num_replicas": 2, "with_gpus": False}
    mlogger = await setup_metric_logger()

    # Spawn services first (triggers registrations via provisioner hook)
    trainer = await TrainActor.options(**service_config).as_service()
    generator = await GeneratorActor.options(**service_config).as_service()

    # Now init config on global (inits backends eagerly across fetchers)
    await mlogger.init_backends.call_one(config)

    for i in range(3):
        print(f"\n=== Global Step {i} ===")
        await trainer.train_step.fanout(i)
        for sub in range(3):
            await generator.generate_step.fanout(i, sub)
        await mlogger.flush.call_one(i)

    # shutdown
    await mlogger.shutdown.call_one()

    await asyncio.gather(
        trainer.shutdown(),
        generator.shutdown(),
    )

    await shutdown()


if __name__ == "__main__":
    asyncio.run(main())
