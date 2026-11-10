import asyncio
import multiprocessing
import os
import time
from types import SimpleNamespace

import pytest

from vla_agent.planner_service import PlannerAdmission, PlannerServiceError


class Log:
    def __init__(self):
        self.events = []

    def event(self, kind, **values):
        self.events.append(dict(kind=kind, **values))


def config(path, slots=1):
    return SimpleNamespace(admission_directory=str(path), admission_slots=slots,
        service_backoff_seconds=.02, admission_wait_seconds=2., review_admission_seconds=.02)


async def test_wait_timeout_cancel_and_release_do_not_leak_slots(tmp_path):
    a, b = PlannerAdmission(config(tmp_path), Log()), PlannerAdmission(config(tmp_path), Log())
    async with a.lease(read_only=False):
        with pytest.raises(PlannerServiceError, match='localAdmissionTimeout'):
            async with b.lease(read_only=True):
                pytest.fail('Occupied slot was granted')

        async def wait():
            async with b.lease(read_only=False):
                pytest.fail('Occupied slot was granted')

        task = asyncio.create_task(wait())
        await asyncio.sleep(.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    async with b.lease(read_only=False):
        pass
    assert sum(e['kind'] == 'planner_admission_acquired' for e in b.log.events) == 1


async def test_shared_backoff_and_quota_are_independent_of_remaining_tokens(tmp_path):
    a, b = PlannerAdmission(config(tmp_path), Log()), PlannerAdmission(config(tmp_path), Log())
    a.failure(PlannerServiceError('serverOverloaded', 'busy'))
    async with b.lease(read_only=False):
        acquired = b.log.events[-1]
        assert acquired['wait_seconds'] >= .015
    a.failure(PlannerServiceError('usageLimitExceeded', 'quota'))
    b.success()
    with pytest.raises(PlannerServiceError, match='usageLimitExceeded'):
        async with b.lease(read_only=False):
            pytest.fail('Quota-blocked model work was admitted')


def worker(path, queue, hold=False):
    async def run():
        gate = PlannerAdmission(config(path, 2), Log())
        async with gate.lease(read_only=False):
            queue.put(('enter', os.getpid(), time.monotonic_ns()))
            if hold:
                await asyncio.sleep(30)
            else:
                await asyncio.sleep(.05)
            queue.put(('leave', os.getpid(), time.monotonic_ns()))
    asyncio.run(run())


def test_cross_process_leases_bound_concurrency_and_survive_process_death(tmp_path):
    context = multiprocessing.get_context('fork')
    queue = context.Queue()
    processes = [context.Process(target=worker, args=(tmp_path, queue)) for _ in range(5)]
    try:
        for p in processes:
            p.start()
        events = [queue.get(timeout=10) for _ in range(10)]
        active, peak = set(), 0
        for kind, pid, _ in sorted(events, key=lambda e: e[2]):
            if kind == 'enter':
                active.add(pid)
                peak = max(peak, len(active))
            else:
                active.remove(pid)
        assert not active and peak <= 2
        for p in processes:
            p.join(timeout=5)
            assert p.exitcode == 0
        victim = context.Process(target=worker, args=(tmp_path, queue, True))
        processes.append(victim)
        victim.start()
        assert queue.get(timeout=10)[0] == 'enter'
        victim.terminate()
        victim.join(timeout=5)
        # Occupy both slots concurrently: the dead process must retain neither.
        async def reacquire():
            a = PlannerAdmission(config(tmp_path, 2), Log())
            b = PlannerAdmission(config(tmp_path, 2), Log())
            async with a.lease(read_only=False), b.lease(read_only=False):
                pass
        asyncio.run(reacquire())
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)
        queue.close()
