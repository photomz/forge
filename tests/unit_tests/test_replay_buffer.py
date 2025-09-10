# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test for data/replay_buffer.py"""

import pytest
import pytest_asyncio
from forge.actors.replay_buffer import ReplayBuffer
from forge.types import Trajectory

from monarch.actor import proc_mesh


class TestReplayBuffer:
    @pytest_asyncio.fixture
    async def replay_buffer(self) -> ReplayBuffer:
        mesh = await proc_mesh(gpus=1)
        replay_buffer = await mesh.spawn(
            "replay_buffer", ReplayBuffer, batch_size=2, max_policy_age=1
        )
        await replay_buffer.setup.call()
        return replay_buffer

    @pytest.mark.asyncio
    async def test_add(self, replay_buffer: ReplayBuffer) -> None:
        trajectory = Trajectory(policy_version=0)
        await replay_buffer.add.call_one(trajectory)
        assert replay_buffer._numel.call_one().get() == 1
        assert replay_buffer._getitem.call_one(0).get() == trajectory
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_add_multiple(self, replay_buffer) -> None:
        trajectory_0 = Trajectory(policy_version=0)
        trajectory_1 = Trajectory(policy_version=1)
        await replay_buffer.add.call_one(trajectory_0)
        await replay_buffer.add.call_one(trajectory_1)
        assert replay_buffer._numel.call_one().get() == 2
        assert replay_buffer._getitem.call_one(0).get() == trajectory_0
        assert replay_buffer._getitem.call_one(1).get() == trajectory_1
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_state_dict_save_load(self, replay_buffer) -> None:
        trajectory = Trajectory(policy_version=0)
        await replay_buffer.add.call_one(trajectory)
        state_dict = replay_buffer.state_dict.call_one().get()
        replay_buffer.clear.call_one().get()
        assert replay_buffer._numel.call_one().get() == 0
        await replay_buffer.load_state_dict.call_one(state_dict)
        assert replay_buffer._numel.call_one().get() == 1
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_evict(self, replay_buffer) -> None:
        trajectory_0 = Trajectory(policy_version=0)
        trajectory_1 = Trajectory(policy_version=1)
        await replay_buffer.add.call_one(trajectory_0)
        await replay_buffer.add.call_one(trajectory_1)
        assert replay_buffer._numel.call_one().get() == 2
        await replay_buffer.evict.call_one(curr_policy_version=2)
        assert replay_buffer._numel.call_one().get() == 1
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_sample(self, replay_buffer) -> None:
        trajectory_0 = Trajectory(policy_version=0)
        trajectory_1 = Trajectory(policy_version=1)
        await replay_buffer.add.call_one(trajectory_0)
        await replay_buffer.add.call_one(trajectory_1)
        assert replay_buffer._numel.call_one().get() == 2

        # Test a simple sampling w/ no evictions
        samples = await replay_buffer.sample.call_one(curr_policy_version=1)
        assert samples is not None
        assert len(samples[0]) == 2

        # Test sampling with overriding batch size
        await replay_buffer.add.call_one(trajectory_0)
        samples = await replay_buffer.sample.call_one(
            curr_policy_version=1, batch_size=1
        )
        assert samples is not None
        assert len(samples[0]) == 1

        # Test sampling w/ overriding batch size (not enough samples in buffer, returns None)
        await replay_buffer.add.call_one(trajectory_0)
        samples = await replay_buffer.sample.call_one(
            curr_policy_version=1, batch_size=3
        )
        assert samples is None
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_sample_with_evictions(self, replay_buffer) -> None:
        trajectory_0 = Trajectory(policy_version=0)
        trajectory_1 = Trajectory(policy_version=1)
        await replay_buffer.add.call_one(trajectory_0)
        await replay_buffer.add.call_one(trajectory_1)
        assert replay_buffer._numel.call_one().get() == 2
        samples = await replay_buffer.sample.call_one(
            curr_policy_version=2, batch_size=1
        )
        assert samples is not None
        assert len(samples[0]) == 1
        assert samples[0][0] == trajectory_1
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_sample_dp_size(self) -> None:
        """Test that len(samples) == dp_size when sampling."""
        mesh = await proc_mesh(gpus=1)
        # Create replay buffer with dp_size=3
        replay_buffer = await mesh.spawn(
            "replay_buffer", ReplayBuffer, batch_size=2, max_policy_age=1, dp_size=3
        )
        await replay_buffer.setup.call()

        # Add enough trajectories to sample
        for i in range(10):
            trajectory = Trajectory(policy_version=0)
            await replay_buffer.add.call_one(trajectory)

        # Sample and verify len(samples) == dp_size
        samples = await replay_buffer.sample.call_one(curr_policy_version=0)
        assert samples is not None
        assert len(samples) == 3  # dp_size
        # Each sub-list should have batch_size samples
        for dp_samples in samples:
            assert len(dp_samples) == 2  # batch_size

        replay_buffer.clear.call_one().get()
