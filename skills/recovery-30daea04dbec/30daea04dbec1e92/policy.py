def run(ctx):
    import numpy as np
    import json

    def decode(value):
        return json.loads(value) if isinstance(value, str) else value

    def compact(value):
        if isinstance(value, dict):
            return {str(k): compact(v) for k, v in value.items() if k not in ('rgb', 'depth_m', 'state')}
        if isinstance(value, np.ndarray):
            return value.tolist() if value.size <= 64 else {'shape': list(value.shape), 'dtype': str(value.dtype)}
        if isinstance(value, (list, tuple)):
            return [compact(v) for v in value] if len(value) <= 64 else {'length': len(value)}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    obs = decode(ctx.observe())
    if ctx.check('task_success')['passed'] is True:
        return 'SUBTASK_DONE'
    if obs['step'] != 1715:
        ctx.note(facts=['Initial image selection is stale; no motion executed.'], hypotheses=[])
        return 'UNRESOLVED'

    with ctx.stage('measure door outward direction') as stage:
        points = []
        for label, box in [('outer door rim', [65, 179, 76, 182]), ('hinge-side rim', [73, 152, 94, 156])]:
            g = decode(ctx.ground_target(target=label, camera='right', observation_step=obs['step'], bbox=box, confidence=0.8))
            measurement = decode(ctx.query_geometry(g['grounding_id']))
            if measurement.get('reliable') is not True or measurement.get('frame') != 'osc_arm_base':
                ctx.note(facts=['Rim geometry unavailable or unreliable.'], hypotheses=[])
                return 'UNRESOLVED'
            if measurement.get('depth_iqr_m', 1.0) > 0.03:
                return 'UNRESOLVED'
            points.append(np.asarray(measurement['centroid_m'], dtype=float))
        outward = points[0] - points[1]
        outward[2] = 0.0
        distance = float(np.linalg.norm(outward))
        if not 0.12 < distance < 0.32:
            return 'UNRESOLVED'
        outward /= distance
        stage.record(outer_rim=points[0].tolist(), hinge_side=points[1].tolist(), outward=outward.tolist())

    initial_pose = obs['eef_control_pose']
    start = np.asarray(initial_pose['position'], dtype=float)
    target_position = start + 0.08 * outward
    moved = 0
    with ctx.stage('withdraw from ineffective inner-opening contact') as stage:
        for index in range(6):
            current = decode(ctx.observe())
            pose = current['eef_control_pose']
            position = np.asarray(pose['position'], dtype=float)
            residual = target_position - position
            error = float(np.linalg.norm(residual))
            if error < 0.008:
                break
            delta = residual * min(1.0, 0.025 / error)
            target = {'position': (position + delta).tolist(),
                      'quaternion_xyzw': pose['quaternion_xyzw'],
                      'gripper_closed': False, 'base_motion': [0.0, 0.0, 0.0, 0.0],
                      'control_mode': 'arm'}
            ctx.move_eef(target, steps=5)
            moved += 5
            after = decode(ctx.observe())
            actual = np.asarray(after['eef_control_pose']['position'], dtype=float)
            stage.record(step=after['step'], measured_position=actual.tolist(), remaining_distance=float(np.linalg.norm(target_position-actual)))
            if ctx.check('task_success')['passed'] is True:
                return 'SUBTASK_DONE'
            if np.linalg.norm(actual-position) < 0.002:
                ctx.note(facts=['Withdrawal produced less than 2 mm measured translation in five feedback steps; stopped further motion.'], hypotheses=['Contact or tracking may restrict outward motion.'])
                break
            if np.linalg.norm(actual-start) > 0.11:
                break

    fresh = decode(ctx.observe())
    ctx.note(facts=['Withdrawal stage finished at observation step ' + str(fresh['step']) + '; executed at most ' + str(moved) + ' action steps.',
                    'Current measured OSC pose: ' + json.dumps(fresh['eef_control_pose'])],
             hypotheses=['A fresh visual selection and calibration inspection are required before grasping or rotating the door; no grasp or closure is claimed.'])
    for camera in ['right', 'left', 'wrist']:
        record = ctx.read_observation(camera)
        ctx.note(facts=['Observation schema and calibration for ' + camera + ': ' + json.dumps(compact(record))[:14000]], hypotheses=[])
    return 'SUBTASK_DONE' if ctx.check('task_success')['passed'] is True else 'UNRESOLVED'
