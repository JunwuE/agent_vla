"""Factor existing toaster policies into RAX-style packages without changing their control rules."""

import argparse
import ast
import copy
import json
import textwrap
from pathlib import Path

from vla_agent.skill_packages import build_package, check_package, execution_bundle
from vla_agent.skills import SkillLibrary

GEOMETRY = 'rgbd-surface-geometry'
CONTROL = 'eef-feedback-control'
PROCEDURE = 'close-toaster-door-from-upright'
SOURCES = [('recovery-6e0569a16adc', '6e0569a16adc2f86'),
           ('recovery-2ff9c7e5cdd3', '2ff9c7e5cdd39598')]


def documentation(name, description, uses, source_refs, procedure, parameters):
    return f'''---
name: {name}
description: {description}
status: draft
robot: PandaOmron
entry: code/run.py
version: 1
uses: {json.dumps([x['skill_id'] for x in uses])}
---
# {name}

## When to use
{description}
Only the measured operating envelopes in the source programs are supported. This refactor is a draft.

## Preconditions
Fresh metric depth, aligned optical-to-OSC-arm-base calibration, and actual measured robot state.
The closing procedure checks its nearly upright door and contact/clearance prerequisites before motion.
Geometry functions do not establish target identity, grasp success or collision freedom.

## Procedure
{procedure}
All commands use the supplied ctx and the existing Harness; no simulator handles or object-state oracle.

## Code
`code/run.py:main(ctx, **params)` returns `{{ok, measured, why}}`.
Python modules use the package name `{name.replace('-', '_')}`. Import dependencies by package name.
{parameters}
Named fixed bounds live in params.py. Scene selections are explicit inputs, never last episode's coordinates.

## Checks
Keep the original entry, depth, alignment, motion, observation and progress checks unchanged.
The host independently checks the whole assigned subtask. A helper's return does not release VLA.

### Perception checkpoints
| Step | Evidence | Check | On failure |
|---|---|---|---|
| Entry | current RGB-D/calibration | selected surfaces and required state are consistent | stop with reason |
| Segments | new observations and actual pose | original tracking/progress bounds | stop; return evidence to Astra |
| Finish | installed task verifier | full subtask success | remain code-owned if unresolved |

## Failure modes
Stale selections, mixed depth, occlusion, lost tracked surfaces, missing calibration, incompatible posture,
blocked motion and insufficient object progress remain possible. Fresh parameters are the first adaptation
when supported; change controller logic only for an observed incompatibility. Do not weaken guards to run.

## Evidence
Source programs: {', '.join(source_refs)}.
Their original logs remain immutable. Source history is provenance, not a fresh physical run of this package.
No promotion to verified is implied. Refactor replay receipts are kept under runs/rax-code-policy-20261031.
'''


def nested_functions(source):
    main = ast.parse(source).body[0]
    return {n.name: n for n in main.body if isinstance(n, ast.FunctionDef)}


def extract(source, node):
    return textwrap.dedent(''.join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno]))


def scaffold(name, description, uses, source_refs, entry, modules, params, procedure, parameters):
    module = name.replace('-', '_')
    return {'SKILL.md': documentation(name, description, uses, source_refs, procedure, parameters),
            'code/run.py': entry, f'code/{module}/params.py': params,
            **{f'code/{module}/{file}': value for file, value in modules.items()}}


def retain(root, name, files, uses, description, contract, origin=None):
    package = build_package(name, files, uses)
    if not check_package(package, files)['ok']:
        raise ValueError(check_package(package, files))
    library = SkillLibrary(root, 'accumulate')
    execution_bundle(package, files, library, 'PandaOmron')
    evidence = dict(kind='format_migration', physical_run=False,
                    source_skill=({k: origin[k] for k in ('id', 'version', 'sha256')} if origin else None),
                    note='Extracted shared functions and explicit parameters; source success is not package verification.')
    item = library.record_candidate(files['code/run.py'], 'PandaOmron', contract.get('completion_condition'),
        evidence, 'format-migration', parent_skill=contract, skill_name=name, description=description,
        when_to_use=origin['when_to_use'] if origin else description, package=package, package_files=files)
    return dict(skill_id=item['id'], version=item['version'])


def migrate(root, output):
    library = SkillLibrary(root)
    sources = [library.get(name, version, 'PandaOmron') for name, version in SOURCES]
    source = sources[0]['code']
    functions = nested_functions(source)
    other = nested_functions(sources[1]['code'])
    shared = ('decode', 'project', 'point_patch', 'pose', 'act', 'record', 'metric')
    for name in shared:
        if ast.dump(functions[name]) != ast.dump(other[name]):
            raise ValueError(f'Cannot share changed function {name} without review')
    refs = [f"{name}/{version}" for name, version in SOURCES]
    geom_source = 'import json\nimport numpy as np\n\n' + '\n\n'.join(
        extract(source, functions[name]) for name in ('decode', 'project', 'point_patch'))
    geometry_files = scaffold(GEOMETRY, 'Project and measure selected RGB-D surface patches using current calibration.', [], refs,
        '''from rgbd_surface_geometry.geometry import project
import numpy as np
def main(ctx, point, intrinsics, camera_to_base):
    pixel = project(np.asarray(point, float), np.asarray(intrinsics, float), np.asarray(camera_to_base, float))
    return {'ok': True, 'measured': {'pixel': pixel.tolist()}, 'why': 'Projection only; no target/contact claim'}
''', {'__init__.py': 'from .geometry import decode, project, point_patch\n', 'geometry.py': geom_source},
        '# Numerical geometry uses current input calibration. Source depth floor: 0.02 m.\n',
        'Pure array computations: decode replies, project base-frame points, unproject finite depth patches.',
        'Inputs are a base-frame 3-vector, current 3x3 intrinsics and current 4x4 optical-to-base transform.')
    geometric_contract = dict(preconditions=[], required_capabilities=[], known_failures=[],
        parameter_sources=['Current aligned camera calibration and measured surface point'],
        parameter_schema=dict(type='object', properties={
            'point': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3},
            'intrinsics': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'minItems': 3, 'maxItems': 3},
            'camera_to_base': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4}, 'minItems': 4, 'maxItems': 4}},
            required=['point', 'intrinsics', 'camera_to_base'], additionalProperties=False))
    geom = retain(root, GEOMETRY, geometry_files, [], 'RGB-D calibrated surface geometry; no robot motion.', geometric_contract)

    pose = extract(source, functions['pose']).replace('def pose():', 'def pose(ctx):')
    act = extract(source, functions['act']).replace('def act(position,quaternion,closed,steps=2):',
        'def move_bounded(ctx, position, quaternion, closed, *, start_step, limit, steps=2):')
    act = act.replace('obs=pose()', 'obs=pose(ctx)').replace('start["step"]', 'start_step')
    act = act.replace('0.020', 'MAX_TRANSLATION_M').replace('0.14', 'MAX_ROTATION_RAD')
    control_source = ('import numpy as np\nfrom scipy.spatial.transform import Rotation\n'
        'from rgbd_surface_geometry import decode\nfrom .params import MAX_TRANSLATION_M, MAX_ROTATION_RAD\n\n'
        + pose + '\n\n' + act)
    ctrl_files = scaffold(CONTROL, 'Bounded OSC EEF command segments from the current measured pose.', [geom], refs,
        '''from eef_feedback_control import move_bounded, pose
def main(ctx, position, quaternion, closed, steps=2):
    start = pose(ctx)
    move_bounded(ctx, position, quaternion, closed, start_step=start['step'], limit=steps, steps=steps)
    current = pose(ctx)
    return {'ok': True, 'measured': {'observation_step': current['step'], 'position': current['eef'].tolist()},
            'why': 'Command segment acknowledged; caller must check the manipulation goal and progress'}
''', {'__init__.py': 'from .control import move_bounded, pose\n', 'control.py': control_source},
        '# Fixed tested command clamps from the source controllers; not tuned for a new scene.\n'
        'MAX_TRANSLATION_M = 0.020\nMAX_ROTATION_RAD = 0.14\n',
        'Read actual pose, clamp translation/rotation, execute a short segment. The caller checks contact/object progress.',
        'position: meters in OSC base; quaternion: xyzw; closed: bool; steps: 1..5. This helper is not a grasp/placement skill.')
    control_contract = dict(preconditions=[], required_capabilities=['osc_eef'], known_failures=[],
        parameter_sources=['Current measured control pose and caller-provided target in OSC base'],
        parameter_schema=dict(type='object', properties={
            'position': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3},
            'quaternion': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4},
            'closed': {'type': 'boolean'}, 'steps': {'type': 'integer', 'minimum': 1, 'maximum': 5}},
            required=['position', 'quaternion', 'closed'], additionalProperties=False))
    ctrl = retain(root, CONTROL, ctrl_files, [geom], 'Measured EEF feedback command helper; goal checks belong to the caller.', control_contract)

    procedures = []
    for old in sources:
        original = old['code']
        fn = nested_functions(original)
        lines = original.splitlines(keepends=True)
        for name in sorted(shared, key=lambda n: fn[n].lineno, reverse=True):
            lines[fn[name].lineno - 1:fn[name].end_lineno] = []
        controller = ''.join(lines).replace('def run(ctx):', 'def execute(ctx, parameters):', 1)
        controller = controller.replace('params = ctx.parameters', 'params = parameters', 1)
        controller = controller.replace('limit = 300', 'limit = ACTION_LIMIT', 1)
        marker = '    limit = ACTION_LIMIT\n'
        if marker not in controller:
            raise ValueError('Source action budget changed; review migration')
        controller = controller.replace(marker, marker + '''    def pose():
        return read_pose(ctx)
    def act(position, quaternion, closed, steps=2):
        return move_bounded(ctx, position, quaternion, closed, start_step=start['step'], limit=limit, steps=steps)
    def record(camera, step):
        return read_camera(ctx, camera, step)
    def metric(label, camera, box, step):
        return measure_surface(ctx, label, camera, box, step)
''', 1)
        controller = ('from rgbd_surface_geometry import decode, project, point_patch\n'
            'from eef_feedback_control import pose as read_pose, move_bounded\n'
            'from .perception import read_camera, measure_surface\nfrom .params import ACTION_LIMIT\n\n' + controller)
        perception = ('import numpy as np\nfrom rgbd_surface_geometry import decode\n\n'
            + extract(original, fn['record']).replace('def record(camera,step):', 'def read_camera(ctx, camera, step):')
            + '\n\n' + extract(original, fn['metric']).replace('def metric(label,camera,box,step):',
                                                               'def measure_surface(ctx, label, camera, box, step):'))
        entry = '''from close_toaster_door_from_upright import main as close_door
def main(ctx, **params):
    return close_door(ctx, **params)
'''
        api = '''from .controller import execute
def main(ctx, **params):
    status = execute(ctx, params)
    observed = ctx.observe()
    return {'ok': status == 'SUBTASK_DONE', 'measured': {'observation_step': observed['step'], 'declared_result': status},
            'why': 'Whole task claimed complete; host rechecks' if status == 'SUBTASK_DONE' else 'Unresolved; inspect stage evidence'}
'''
        files = scaffold(PROCEDURE, old['when_to_use'], [geom, ctrl], [f"{old.get('storage_id',old['id'])}/{old['version']}"],
            entry, {'__init__.py': api, 'controller.py': controller, 'perception.py': perception},
            '# Source action budget retained. Host may impose a lower remaining budget.\nACTION_LIMIT = 300\n',
            'Check nearly upright geometry; release, clear, withdraw/align, recontact, seat and verify using source stages.',
            'The original parameter_schema is retained unchanged; image boxes/observation_step and supported cameras are refreshed per invocation.')
        contract = copy.deepcopy(old)
        new = retain(root, PROCEDURE, files, [geom, ctrl], 'Close the toaster oven door. | Modular near-upright recovery with fresh measurements.', contract, old)
        procedures.append(dict(source={k: old[k] for k in ('id', 'version', 'sha256')}, migrated=new))
    result = dict(geometry=geom, control=ctrl, procedures=procedures,
                  note='Draft packages; source success is not a new physical success. Original files unchanged.')
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=Path('skills'))
    parser.add_argument('--output', type=Path, default=Path('runs/rax-code-policy-20261031/migration.json'))
    args = parser.parse_args()
    print(json.dumps(migrate(args.library, args.output)))
