from .controller import execute
def main(ctx, **params):
    status = execute(ctx, params)
    observed = ctx.observe()
    return {'ok': status == 'SUBTASK_DONE', 'measured': {'observation_step': observed['step'], 'declared_result': status},
            'why': 'Whole task claimed complete; host rechecks' if status == 'SUBTASK_DONE' else 'Unresolved; inspect stage evidence'}
