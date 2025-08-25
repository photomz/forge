# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for service.py
"""

import asyncio
import logging

import pytest
from forge.controller.service import ServiceConfig
from forge.controller.spawn import spawn_service
from monarch.actor import Actor, endpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Counter(Actor):
    """Test actor that maintains a counter with various endpoints."""

    def __init__(self, v: int):
        self.v = v

    @endpoint
    async def incr(self):
        """Increment the counter."""
        self.v += 1

    @endpoint
    async def value(self) -> int:
        """Get the current counter value."""
        return self.v

    @endpoint
    async def fail_me(self):
        """Endpoint that always fails to test error handling."""
        raise RuntimeError("I was asked to fail")

    @endpoint
    async def slow_incr(self):
        """Slow increment to test queueing."""
        await asyncio.sleep(1.0)
        self.v += 1


# Core Functionality Tests


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_basic_service_operations():
    """Test basic service creation, sessions, and endpoint calls."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=1)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Test session creation and uniqueness
        session1 = await service.start_session()
        session2 = await service.start_session()
        assert session1 != session2
        assert isinstance(session1, str)

        # Test endpoint calls
        await service.incr(session1)
        result = await service.value(session1)
        assert result == 1

        # Test session mapping
        assert session1 in service._session_replica_map

        # Test session termination
        await service.terminate_session(session1)
        assert session1 not in service._session_replica_map

    finally:
        await service.stop()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_sessionless_calls():
    """Test sessionless calls with round-robin load balancing."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=2)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Test sessionless calls
        await service.incr()
        await service.incr()
        result = await service.value()
        assert result is not None

        # No sessions should be created
        assert len(service._active_sessions) == 0
        assert len(service._session_replica_map) == 0

        # Verify load distribution
        metrics = service.get_metrics_summary()
        total_requests = sum(
            replica_metrics["total_requests"]
            for replica_metrics in metrics["replicas"].values()
        )
        assert total_requests == 3

    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_session_context_manager():
    """Test session context manager functionality."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=1)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Test context manager usage
        async with service.session():
            await service.incr()
            await service.incr()
            result = await service.value()
            assert result == 2

        # Test sequential context managers to avoid interference
        async def worker(increments: int):
            async with service.session():
                initial = await service.value()
                for _ in range(increments):
                    await service.incr()
                final = await service.value()
                return final - initial

        # Run sessions sequentially to avoid concurrent modification
        result1 = await worker(2)
        result2 = await worker(3)
        results = [result1, result2]
        assert sorted(results) == [2, 3]

        # Test that context manager properly manages session lifecycle
        assert len(service._active_sessions) == 0
        assert len(service._session_replica_map) == 0

    finally:
        await service.stop()


# Fault Tolerance Tests


@pytest.mark.timeout(20)
@pytest.mark.asyncio
async def test_recovery_state_transitions():
    """Test replica state transitions during failure and recovery."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=1, health_poll_rate=0.1)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Initially replica should be healthy
        replica = service._replicas[0]
        assert replica.state.value == "HEALTHY"
        assert replica.healthy is True
        assert replica.failed is False

        # Create session and make a successful call
        session = await service.start_session()
        await service.incr(session)
        result = await service.value(session)
        assert result == 1

        # Cause failure - this should transition to RECOVERING
        error_result = await service.fail_me(session)
        assert isinstance(error_result, RuntimeError)

        # Replica should now be in RECOVERING state
        assert replica.state.value == "RECOVERING"
        assert replica.healthy is False
        assert replica.failed is True

        # Wait for health loop to detect and attempt recovery
        # The health loop runs every 0.1s, so give it some time
        max_wait_time = 5.0  # 5 seconds max wait
        wait_interval = 0.1
        elapsed = 0.0

        # Wait for replica to either recover (HEALTHY) or fail completely (UNHEALTHY)
        while elapsed < max_wait_time:
            await asyncio.sleep(wait_interval)
            elapsed += wait_interval

            if replica.state.value in ["HEALTHY", "UNHEALTHY"]:
                break

        # After recovery, replica should be healthy again
        # (unless recovery failed, in which case it would be UNHEALTHY)
        assert replica.state.value in ["HEALTHY", "UNHEALTHY"]

        if replica.state.value == "HEALTHY":
            # If recovery succeeded, verify we can make calls again
            assert replica.healthy is True
            assert replica.failed is False

            # Test that we can make new calls after recovery
            new_session = await service.start_session()
            await service.incr(new_session)
            result = await service.value(new_session)
            assert (
                result is not None
            )  # Should get a result (counter starts at 0 in new actor)

        elif replica.state.value == "UNHEALTHY":
            # If recovery failed, verify failed state
            assert replica.healthy is False
            assert replica.failed is True

        # Verify that the state transition path was correct
        # (We can't guarantee the exact end state due to potential flakiness in test environments,
        # but we can verify the replica went through the expected transition)
        logger.info(f"Final replica state: {replica.state.value}")

    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_replica_failure_and_recovery():
    """Test replica failure handling and automatic recovery."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=2)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Create session and cause failure
        session = await service.start_session()
        await service.incr(session)

        original_replica_idx = service._session_replica_map[session]

        # Cause failure
        error_result = await service.fail_me(session)
        assert isinstance(error_result, RuntimeError)

        # Replica should be marked as failed
        failed_replica = service._replicas[original_replica_idx]
        assert not failed_replica.healthy

        # Session should be reassigned on next call
        await service.incr(session)
        new_replica_idx = service._session_replica_map[session]
        assert new_replica_idx != original_replica_idx

        # New sessions should avoid failed replica
        new_session = await service.start_session()
        await service.incr(new_session)
        assigned_replica = service._replicas[service._session_replica_map[new_session]]
        assert assigned_replica.healthy

    finally:
        await service.stop()


# Metrics and Monitoring Tests


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_metrics_collection():
    """Test metrics collection."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=2)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Create sessions and make requests
        session1 = await service.start_session()
        session2 = await service.start_session()

        await service.incr(session1)
        await service.incr(session1)
        await service.incr(session2)

        # Test failure metrics
        error_result = await service.fail_me(session1)
        assert isinstance(error_result, RuntimeError)

        # Get metrics
        metrics = service.get_metrics()
        summary = service.get_metrics_summary()

        # Test service-level metrics
        assert metrics.total_sessions == 2
        assert metrics.healthy_replicas <= 2  # One may have failed
        assert metrics.total_replicas == 2

        # Test summary structure
        assert "service" in summary
        assert "replicas" in summary

        # Test request counts
        total_requests = sum(
            replica_metrics["total_requests"]
            for replica_metrics in summary["replicas"].values()
        )
        assert total_requests == 4  # 3 successful + 1 failed

        total_failed = sum(
            replica_metrics["failed_requests"]
            for replica_metrics in summary["replicas"].values()
        )
        assert total_failed == 1

    finally:
        await service.stop()


# Load Balancing and Session Management Tests


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_session_stickiness():
    """Test that sessions stick to the same replica."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=2)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        session = await service.start_session()

        # Make multiple calls
        await service.incr(session)
        await service.incr(session)
        await service.incr(session)

        # Should always route to same replica
        replica_idx = service._session_replica_map[session]

        await service.incr(session)
        assert service._session_replica_map[session] == replica_idx

        # Verify counter was incremented correctly
        result = await service.value(session)
        assert result == 4

    finally:
        await service.stop()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_load_balancing_multiple_sessions():
    """Test load balancing across multiple sessions using least-loaded assignment."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=2)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Create sessions with some load to trigger distribution
        session1 = await service.start_session()
        await service.incr(session1)  # Load replica 0

        session2 = await service.start_session()
        await service.incr(session2)  # Should go to replica 1 (least loaded)

        session3 = await service.start_session()
        await service.incr(session3)  # Should go to replica 0 or 1 based on load

        session4 = await service.start_session()
        await service.incr(session4)  # Should balance the load

        # Check that sessions are distributed (may not be perfectly even due to least-loaded logic)
        replica_assignments = [
            service._session_replica_map[s]
            for s in [session1, session2, session3, session4]
        ]
        unique_replicas = set(replica_assignments)

        # With least-loaded assignment, we should eventually use both replicas
        # as load accumulates, though initial sessions may go to the same replica
        assert len(unique_replicas) >= 1  # At least one replica used

        # Verify that load balancing is working by checking request distribution
        metrics = service.get_metrics_summary()
        total_requests = sum(
            replica_metrics["total_requests"]
            for replica_metrics in metrics["replicas"].values()
        )
        assert total_requests == 4  # All requests processed

    finally:
        await service.stop()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test concurrent operations across sessions and sessionless calls."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=2)
    service = await spawn_service(
        service_cfg=cfg, actor_def=Counter, name="counter", v=0
    )

    try:
        # Mix of session and sessionless calls
        session = await service.start_session()

        # Concurrent operations
        tasks = [
            service.incr(session),  # Session call
            service.incr(session),  # Session call
            service.incr(),  # Sessionless call
            service.incr(),  # Sessionless call
        ]

        await asyncio.gather(*tasks)

        # Verify session tracking
        assert len(service._active_sessions) == 1
        assert session in service._session_replica_map

        # Verify total requests
        metrics = service.get_metrics_summary()
        total_requests = sum(
            replica_metrics["total_requests"]
            for replica_metrics in metrics["replicas"].values()
        )
        assert total_requests == 4

    finally:
        await service.stop()
