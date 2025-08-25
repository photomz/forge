# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Factory-based service spawning for the Monarch rollout system."""

import logging
from typing import Type

from monarch.actor import Actor, proc_mesh

from forge.controller import Service, ServiceConfig
from forge.controller.interface import ServiceInterface

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


async def spawn_service(
    service_cfg: ServiceConfig, actor_def: Type[Actor], *actor_args, **actor_kwargs
) -> ServiceInterface:
    """Spawns a service based on the actor class.

    Args:
        service_cfg: Service configuration
        actor_def: Actor class definition
        *actor_args: Arguments to pass to actor constructor
        **actor_kwargs: Keyword arguments to pass to actor constructor

    Returns:
        A ServiceInterface that provides access to the Service Actor
    """
    # Create a single-node proc_mesh and actor_mesh for the Service Actor
    logger.info("Spawning Service Actor for %s", actor_def.__name__)
    m = await proc_mesh(gpus=1)
    service_actor = await m.spawn(
        "service", Service, service_cfg, actor_def, actor_args, actor_kwargs
    )
    await service_actor.__initialize__.call_one()

    # Return the ServiceInterface that wraps the proc_mesh, actor_mesh, and actor_def
    return ServiceInterface(m, service_actor, actor_def)
