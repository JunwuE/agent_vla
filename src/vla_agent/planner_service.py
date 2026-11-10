"""Cross-process model admission and typed provider failures; no robot authority."""

import asyncio
import fcntl
import json
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path


class PlannerServiceError(RuntimeError):
    def __init__(self, code, message):
        self.code = next(iter(code), 'unknown') if isinstance(code, dict) else str(code or 'unknown')
        self.message = str(message or 'No service error message')[:1000]
        super().__init__(f'Codex model request failed [{self.code}]: {self.message}')

    @property
    def retryable(self):
        return self.code in {'serverOverloaded', 'responseStreamDisconnected', 'httpConnectionFailed'}

    @property
    def reason(self):
        return {'serverOverloaded': 'planner_service_overloaded',
                'usageLimitExceeded': 'planner_usage_limit',
                'contextWindowExceeded': 'planner_context_limit',
                'localAdmissionTimeout': 'planner_admission_timeout'}.get(self.code, 'planner_service_error')


class PlannerAdmission:
    """flock leases release on cancellation/process exit, without stale PID leases."""

    def __init__(self, config, log):
        self.config, self.log = config, log
        self.directory = Path(config.admission_directory) if config.admission_directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            with self.state() as state:
                previous = state.setdefault('slots', config.admission_slots)
                if previous != config.admission_slots:
                    raise ValueError('Shared admission directory has a different slot count')

    @contextmanager
    def state(self):
        lock = (self.directory / 'state.lock').open('a+')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.directory / 'state.json'
            value = json.loads(path.read_text()) if path.exists() else {}
            before = json.dumps(value, sort_keys=True)
            yield value
            if json.dumps(value, sort_keys=True) != before:
                temporary = self.directory / f'state-{uuid.uuid4().hex}.tmp'
                temporary.write_text(json.dumps(value))
                temporary.replace(path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

    def failure(self, error):
        if not self.directory:
            return
        with self.state() as state:
            if error.code == 'usageLimitExceeded':
                state['quota_blocked'] = True
            if error.retryable:
                failures = state.get('consecutive_failures', 0) + 1
                state['consecutive_failures'] = failures
                delay = min(self.config.service_backoff_seconds * 2 ** min(failures - 1, 6), 60)
                state['cooldown_until'] = max(state.get('cooldown_until', 0), time.time() + delay)
            state['last_error'] = error.code
        self.log.event('planner_service_backoff', code=error.code,
                       cooldown_until=state.get('cooldown_until'), quota_blocked=state.get('quota_blocked', False))

    def success(self):
        if self.directory:
            with self.state() as state:
                state['consecutive_failures'] = 0
                # An in-flight success must not clear another request's quota block or cooldown.

    @asynccontextmanager
    async def lease(self, *, read_only):
        if self.directory is None:
            yield
            return
        started = time.monotonic()
        limit = self.config.review_admission_seconds if read_only else self.config.admission_wait_seconds
        handle, slot = None, None
        identity = uuid.uuid4().hex
        self.log.event('planner_admission_wait', lease_id=identity, read_only=read_only, timeout_seconds=limit)
        try:
            while handle is None:
                with self.state() as state:
                    if state.get('quota_blocked'):
                        raise PlannerServiceError('usageLimitExceeded', 'Shared provider quota block; no new model request sent')
                    eligible = time.time() >= state.get('cooldown_until', 0)
                    if eligible:
                        for index in range(self.config.admission_slots):
                            candidate = (self.directory / f'slot-{index}.lock').open('a+')
                            try:
                                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                candidate.close()
                                continue
                            handle, slot = candidate, index
                            break
                if handle is not None:
                    break
                if time.monotonic() - started >= limit:
                    raise PlannerServiceError('localAdmissionTimeout', 'No model slot available within the bounded admission wait')
                await asyncio.sleep(min(.1, max(.001, limit - (time.monotonic() - started))))
            self.log.event('planner_admission_acquired', lease_id=identity, slot=slot,
                           directory=str(self.directory), wait_seconds=time.monotonic() - started)
            yield
        finally:
            if handle is not None:
                try:
                    self.log.event('planner_admission_released', lease_id=identity, slot=slot)
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                    handle.close()
