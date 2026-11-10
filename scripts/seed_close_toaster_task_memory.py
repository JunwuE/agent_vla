"""Consolidate the first bounded toaster task guide from exact historical episode evidence."""

import argparse
import json
from pathlib import Path

from vla_agent.experience import ExperienceStore
from vla_agent.task_memory import TaskMemoryStore, TaskProfile, version_of


def build(experiences):
    scope = dict(robot='PandaOmron', environment_kind='robocasa', task_name='CloseToasterOvenDoor')

    def ref(episode_id, role='context'):
        record = experiences.get('PandaOmron', episode_id, scope=scope)
        return dict(episode_id=episode_id, evidence_sha256=record['evidence_sha256'], role=role)

    normal = ref('cf775f187a554919a5668ab347f7bbbf', 'counterexample')  # seed 8, no code recovery
    arc = ref('765d307e499e43bfab1fe2a5997ec423', 'support')  # seed 9, original arc controller
    clearance = ref('c0494b23d17947f99ea55b94172de2e1', 'support')  # seed 44, completed during clearance
    failed = ref('55f019b8378346378464066a10acec48', 'context')  # seed 36, failed episode
    phases = [
        dict(id='approach_align', goal='Approach and align with the visible toaster door.',
             progress_signals=['Measured approach and orientation changes suit the current door and gripper geometry.'],
             completion_signals=['Current observations establish an appropriate manipulation pose.'],
             failure_signals=['Repeated alignment attempts show a specific persistent obstruction or wrong target.']),
        dict(id='engage_raise', goal='Establish interaction and raise the hinged door.',
             prerequisites=['Current engagement geometry supports the intended interaction.'],
             progress_signals=['Door motion agrees with the intended rotation; monitor loss of engagement.'],
             completion_signals=['The door reaches a measured nearly upright state.'],
             failure_signals=['Repeated manipulation has no object effect, or engagement is lost.']),
        dict(id='clear_and_seat', goal='If needed, establish clearance and seat the nearly upright door.',
             prerequisites=['Current geometry satisfies the selected procedure\'s entry conditions.'],
             progress_signals=['Measured clearance, tracking and door motion match the stage being executed.'],
             completion_signals=['The door appears seated; request the installed whole-task verifier.'],
             failure_signals=['Occluded/mixed-depth surfaces, blocked clearance or inconsistent tracking.']),
        dict(id='verify', goal='Confirm the complete assigned task.',
             progress_signals=['Observe the resulting state and independent task verification.'],
             completion_signals=['Installed task_success passes.'],
             failure_signals=['The door remains ajar/reopens or the installed verifier does not pass.']),
    ]
    rules = [
        dict(id='preserve_effective_vla', when=['First attempt or continued execution has phase-consistent progress.',
             'No current evidence of a persistent mismatch, wrong target or unsafe state.'],
             recommendation='continue_vla', status='observed_pattern',
             rationale='The same task has completed without code recovery. A historical recovery does not prove the current attempt needs takeover.',
             tools=['read_execution', 'evaluate_subtask'],
             avoid=['Do not interrupt merely because another seed needed code, or rewrite language into a motion tutorial.'], evidence=[normal]),
        dict(id='diagnose_before_handoff', when=['Repeated interaction may be ineffective.',
             'It is unclear whether the robot is still approaching, aligning or manipulating the door.'],
             recommendation='inspect', status='hypothesis',
             rationale='Compare phase-specific expected object change with separated observations and measured motion; completed and failed episodes are both relevant counterchecks.',
             tools=['read_execution', 'measure_targets', 'evaluate_subtask'],
             avoid=['Low EEF translation or a low Flora score alone does not establish a failed manipulation.'],
             evidence=[normal, failed]),
        dict(id='reuse_nearly_upright_closure', when=['Astra has confirmed recovery is needed.',
             'Fresh geometry establishes a nearly upright door and the selected version\'s posture, visibility and clearance requirements.'],
             recommendation='consider_skill', status='observed_pattern',
             rationale='Original controllers completed the seed 9 and seed 44 episodes. Their modular descendants match recorded requests, but have no new physical validation. Seed 44 completed during clearance, not all later seating branches.',
             tools=['list_skill_versions', 'read_skill', 'measure_targets', 'check_skill'],
             skills=[dict(skill_id='close-toaster-door-from-upright', version='b22b65f09b8cc992'),
                     dict(skill_id='close-toaster-door-from-upright', version='90a28a7feeb1d020')],
             avoid=['Do not apply this entry to a fully lowered door or claim a general grasp skill.',
                    'New seed, image boxes and observation_step require fresh parameters before considering a source patch.'],
             evidence=[arc, clearance, failed]),
        dict(id='check_clearance_before_release', when=['A code procedure proposes opening fingers to disengage or release.',
             'Current contact, door support or exit clearance is uncertain.'],
             recommendation='inspect', status='hypothesis',
             rationale='Release is a conditional operation. Recheck the selected procedure\'s geometry and observe the effect; the successful clearance history does not guarantee a different contact state is safe.',
             tools=['observe', 'measure_targets', 'read_skill'],
             avoid=['Do not treat set_gripper(open) as evidence that release, support or the task succeeded.'],
             evidence=[arc, clearance]),
    ]
    return TaskProfile(scope=scope,
        summary='CloseToasterOvenDoor guidance distilled from recorded episodes: preserve effective VLA execution, diagnose by phase, and reuse checked near-upright closure code with fresh geometry.',
        applicability=['PandaOmron in RoboCasa365 with measured OSC arm-base pose and aligned metric RGB-D.',
                       'Historical results used evolving policies and layouts; they are not a fixed-version success-rate estimate.',
                       'Phase suggestions are not executable subtasks or VLA language. Astra must select current applicable phases.'],
        phase_guide=phases, rules=rules,
        open_questions=['New modular packages still need physical tests on matching entry states and new layouts.',
                        'A reusable whole procedure from the fully lowered door is not established by the near-upright skills.',
                        'Release/support checks need task-specific observable evidence; no contact-force oracle is available.'])


def main(args):
    experiences = ExperienceStore(args.experiences)
    profile = build(experiences)
    store = TaskMemoryStore(args.output)
    current = next((x for x in store.index.values() if x['profile']['scope'] == profile.scope.model_dump()), None)
    if current and current['version'] != version_of(profile.model_dump()):
        raise ValueError('This task guide was revised; do not overwrite it with the initial seed profile')
    receipt = store.write_profile(profile, experiences, expected_version=current['version'] if current else None)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiences', type=Path, default=Path('memory/close_toaster/experiences'))
    parser.add_argument('--output', type=Path, default=Path('memory/close_toaster/tasks'))
    main(parser.parse_args())
