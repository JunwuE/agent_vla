"""Async JSON-RPC client for an owned Codex app-server process."""

import asyncio
import json
import os
import signal

from .data import dumps


class CodexTransport:
    def __init__(self):
        self.process = None
        self.reader = None
        self.pending = {}
        self.events = asyncio.Queue()
        self.next_id = 0
        self.stderr = None

    async def start(self, argv, cwd):
        self.stderr = (cwd / "stderr.log").open("ab")
        self.process = await asyncio.create_subprocess_exec(
            *argv, cwd=cwd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=self.stderr, start_new_session=True, limit=16 << 20,
        )
        self.reader = asyncio.create_task(self._read())

    async def _read(self):
        error = EOFError("Codex app-server closed; inspect codex/stderr.log")
        try:
            while line := await self.process.stdout.readline():
                message = json.loads(line)
                if "id" in message and "method" not in message:
                    future = self.pending.get(message["id"])
                    if future is not None and not future.done():
                        future.set_result(message)
                else:
                    await self.events.put(message)
        except (ValueError, OSError) as exc:
            error = exc
        finally:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(error)
            await self.events.put(error)

    async def send(self, message):
        if self.process is None or self.process.returncode is not None:
            raise RuntimeError("Codex app-server is unavailable")
        self.process.stdin.write((dumps(message) + "\n").encode())
        await self.process.stdin.drain()

    async def request(self, method, params, timeout=30):
        self.next_id += 1
        identifier = self.next_id
        future = asyncio.get_running_loop().create_future()
        self.pending[identifier] = future
        try:
            await self.send(dict(jsonrpc="2.0", id=identifier, method=method, params=params))
            async with asyncio.timeout(timeout):
                response = await future
            if "error" in response:
                raise RuntimeError(f"Codex {method}: {response['error']}")
            return response["result"]
        finally:
            self.pending.pop(identifier, None)

    async def notify(self, method, params):
        await self.send(dict(jsonrpc="2.0", method=method, params=params))

    async def reply(self, identifier, result):
        await self.send(dict(jsonrpc="2.0", id=identifier, result=result))

    async def next_event(self):
        event = await self.events.get()
        if isinstance(event, BaseException):
            raise event
        return event

    async def close(self):
        if self.process and self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                await asyncio.wait_for(self.process.wait(), 3)
            except TimeoutError:
                os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()
            except ProcessLookupError:
                pass
        if self.reader:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
            self.reader = None
        if self.stderr:
            self.stderr.close()
            self.stderr = None
