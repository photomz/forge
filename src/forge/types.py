# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, Union


class Message(TypedDict):
    role: str
    content: str | dict[str, Any]
    tools: dict[str, Any] | None


@dataclass
class ForgeEnvInfo:
    """Environment info returned with observations."""

    episode_id: str | None = None
    step_count: int = 0
    metadata: dict | None = None


@dataclass(kw_only=True)
class Observation:
    """Base class for environment observations.

    Contract:
    - Should contain all information needed by an agent to make decisions
    - Should be serializable/deserializable
    - Should be immutable (or treated as such)
    Args:
        done: Whether the episode/conversation is complete
        reward: Optional reward signal (can be boolean, int, or float)
        metadata: Additional data that doesn't affect agent decisions but may be useful
                 for transforms, logging, evaluation, etc.
    """

    done: bool = False
    reward: bool | int | float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class Action:
    """Base class for environment actions.

    Contract:
    - Should contain all information needed to execute a step in the environment
    - Should be serializable/deserializable
    - Should be immutable (or treated as such)

    Args:
        metadata: Additional data that may be useful for logging, debugging, or transforms
    """

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """A trajectory containing a sequence of states, actions, etc."""

    policy_version: int
    states: list[Observation] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)

    def __post_init__(self):
        assert self.policy_version >= 0


@dataclass(kw_only=True)
class State:
    """Base class for environment state.

    Contract:
    - Should contain all information needed to restore the environment
    - Should be serializable/deserializable
    - May contain information not exposed in observations

    Args:
        metadata: Additional state information that may be useful for debugging or analysis
    """

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessConfig:
    """A proc_mesh config for the torchx scheduler."""

    scheduler: Literal["mast", "local"] = "local"
    num_procs: int = 1
    with_gpus: bool = False
    num_hosts: int = 1
    # The following is mast specific.
    oncall: str = "torchtune"
    identity: str = "pytorch_distributed"
    image: str = "forge_workspace:latest"


@dataclass
class ServiceConfig:
    """A service config."""

    procs_per_replica: int
    num_replicas: int
    with_gpus: bool = False
    num_hosts: int = 1
    scheduler: Literal["mast", "local"] = "local"
    oncall: str = "torchtune"
    identity: str = "pytorch_distributed"
    image: str = "forge_workspace:latest"
    # ServiceConfig-specific fields
    health_poll_rate: float = 0.2
    replica_max_concurrent_requests: int = 10
    return_first_rank_result: bool = (
        True  # Whether or not to auto-unwrap ValueMesh to first rank's result
    )

    def to_process_config(self) -> ProcessConfig:
        """Extract ProcessConfig from this ServiceConfig.
        Maps procs_per_replica to num_procs for ProcessConfig.
        """
        return ProcessConfig(
            scheduler=self.scheduler,
            num_procs=self.procs_per_replica,
            with_gpus=self.with_gpus,
            num_hosts=self.num_hosts,
            oncall=self.oncall,
            identity=self.identity,
            image=self.image,
        )


Scalar = Union[int, float]
