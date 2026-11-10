from eef_feedback_control import move_bounded, pose
def main(ctx, position, quaternion, closed, steps=2):
    start = pose(ctx)
    move_bounded(ctx, position, quaternion, closed, start_step=start['step'], limit=steps, steps=steps)
    current = pose(ctx)
    return {'ok': True, 'measured': {'observation_step': current['step'], 'position': current['eef'].tolist()},
            'why': 'Command segment acknowledged; caller must check the manipulation goal and progress'}
