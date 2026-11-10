"""Diagnostic intervention injection with real RoboCasa/pi0.5, no reasoning model calls.

Run against a frozen source via PYTHONPATH. The synthetic request is a control
experiment, not a genuine failure annotation or a new supervised success-rate run.
"""

import argparse
import asyncio
import json
from pathlib import Path

from vla_agent.adapters.pi05 import Pi05
from vla_agent.adapters.robocasa import RoboCasaEnvironment
from vla_agent.config import load_config
from vla_agent.contracts import Decision, DecisionLabel, ExecutionReview, Intervention
from vla_agent.data import Recorder
from vla_agent.runtime import AgentRuntime


class DiagnosticPlanner:
    calls = 0

    async def plan(self, decision, context, toolkit):
        if context['intent'] == 'review_running':
            return ExecutionReview(action='request_intervention', assessment='failed_attempt',
                phase='diagnostic_injection', reason='Synthetic pause request to test chunk preservation',
                evidence=[dict(observation_step=context['binding'].observation_step,
                               fact='Diagnostic only: injected at the configured step, not a failure label')])
        await toolkit.call('observe', {})
        return Intervention(kind='continue', reason='Diagnostic read-only confirmation; keep task unchanged')

    async def close(self):
        pass


class DiagnosticRuntime(AgentRuntime):
    injected = False

    async def poll_review(self):
        if not self.injected and self.harness.obs.step == self.injection_step:
            self.injected = True
            decision = Decision(binding=self.memory.binding(self.harness.epoch),
                                label=DecisionLabel.ESCALATE, trigger='diagnostic_readonly_pause')
            self.log.event('diagnostic_pause_injected', step=self.injection_step,
                           note='Synthetic request; no Astra inference or learned failure label')
            self.reviewer.submit(decision, self.harness, self.memory)
            await self.reviewer.pending
        return await super().poll_review()


async def main(args):
    config = load_config(args.config)
    config.runtime.supervision_enabled = False
    config.runtime.initial_planning = False
    config.runtime.memory_mode = 'frozen'
    config.runtime.experiment_label = 'diagnostic-readonly-pause-sequence'
    config.planner.kind = 'mock'
    log = Recorder(args.output, config)
    try:
        runtime = DiagnosticRuntime(config, RoboCasaEnvironment(config.environment, args.output),
                                    Pi05(config.vla, log), None, DiagnosticPlanner(), log)
        runtime.injection_step = args.step
        result = await runtime.run()
        print(json.dumps(result), flush=True)
        if result['error']:
            raise SystemExit(1)
    finally:
        log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--step', type=int, default=532)
    asyncio.run(main(parser.parse_args()))
