"""Task contracts, not a fixed plan. Astra selects their order and executor.

These optional checks use simulator state, expose only booleans, and must be
reported as verifier-assisted evaluation. They never supply object coordinates.
"""

from ..contracts import Subtask


def subtask_catalog(task_name):
    if task_name == 'PrepareCoffee':
        return [
            Subtask(name='retrieve_mug',
                    instruction='Pick the mug from the cabinet and place it on the counter.',
                    completion_condition='coffee_mug_on_counter'),
            Subtask(name='position_mug',
                    instruction='Pick the mug from the counter and place it under the coffee machine dispenser.',
                    preconditions=['coffee_mug_on_counter'], completion_condition='coffee_mug_ready'),
            Subtask(name='start_coffee', instruction='Press the button on the coffee machine to serve coffee.',
                    preconditions=['coffee_mug_ready'], completion_condition='task_success'),
        ]
    if task_name != 'WaffleReheat':
        raise ValueError(f'Native subtask checks not implemented for {task_name}')
    return [
        Subtask(name='open_microwave', instruction='Open the microwave door.',
                completion_condition='microwave_open'),
        Subtask(name='load_waffle_bowl', instruction='Place the bowl with waffle inside the microwave.',
                preconditions=['microwave_open', 'waffle_in_bowl'],
                completion_condition='waffle_bowl_loaded'),
        Subtask(name='close_loaded_microwave', instruction='Close the microwave door.',
                preconditions=['waffle_bowl_loaded'], completion_condition='loaded_microwave_closed'),
        Subtask(name='start_microwave', instruction='Press the start button on the microwave.',
                preconditions=['loaded_microwave_closed'], completion_condition='task_success'),
    ]


def native_checks(native, task_name):
    subtask_catalog(task_name)  # Fail explicitly for unsupported tasks.
    from robocasa.utils import object_utils as ou

    if task_name == 'PrepareCoffee':
        released = bool(ou.gripper_obj_far(native, 'obj'))
        ready = bool(native.coffee_machine.check_receptacle_placement_for_pouring(native, 'obj'))
        on_counter = bool(ou.check_obj_any_counter_contact(native, 'obj'))
        outside_cabinet = not bool(ou.obj_inside_of(native, 'obj', native.cab))
        return dict(coffee_mug_on_counter=on_counter and outside_cabinet and released,
                    coffee_mug_ready=ready and released)

    food = bool(ou.check_obj_in_receptacle(native, 'waffle', 'waffle_container'))
    inside = bool(ou.obj_inside_of(native, 'waffle_container', native.microwave))
    # A release/retreat check makes the placement boundary safe for door motion.
    released = bool(ou.gripper_obj_far(native, 'waffle_container'))
    opened = bool(native.microwave.is_open(native))
    closed = bool(native.microwave.is_closed(native))
    loaded = food and inside and released
    return dict(microwave_open=opened, waffle_in_bowl=food,
                waffle_bowl_loaded=loaded, loaded_microwave_closed=loaded and closed)
