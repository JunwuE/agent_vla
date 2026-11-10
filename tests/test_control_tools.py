import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation
from test_runtime import ROOT, make_harness

from vla_agent.adapters.mock import MockEnvironment, MockVLA
from vla_agent.adapters.robocasa_control import (
    read_fingerpad_geometry,
    read_gripper_kinematics,
    read_robot_control,
)
from vla_agent.contracts import Source
from vla_agent.control import ActionEdit, EEFRequest, EEFTarget, OSCControl, execute_target
from vla_agent.data import read_events
from vla_agent.harness import StaleControl
from vla_agent.memory import SkillLibrary
from vla_agent.recovery import RecoveryRunner
from vla_agent.toolkit import ImageResult, PlannerToolkit


def test_gripper_axis_accounts_for_finger_body_rotation_and_rejects_unknown_grippers():
    site_rotation = Rotation.from_euler("xyz", [0.4, -0.3, 0.7]).as_matrix()
    finger_rotation = site_rotation @ Rotation.from_euler("z", np.pi / 2).as_matrix()
    model = SimpleNamespace(joint_name2id=lambda name: int(name[-1]), jnt_type=[2, 2],
                            jnt_bodyid=[0, 1], jnt_axis=np.array([[0, 1, 0], [0, 1, 0]]))
    data = SimpleNamespace(site_xmat=np.array([site_rotation.reshape(9)]),
                           body_xmat=np.array([finger_rotation.reshape(9)] * 2))
    robot = SimpleNamespace(gripper={"right": SimpleNamespace(joints=["finger0", "finger1"])},
                            eef_site_id={"right": 0})
    sim = SimpleNamespace(model=model, data=data)
    result = read_gripper_kinematics(robot, sim, 12)
    assert result["available"] and result["step"] == 12
    np.testing.assert_allclose(result["jaw_separation_axis_eef"], [-1, 0, 0], atol=1e-12)
    model.jnt_type[0] = 3
    assert not read_gripper_kinematics(robot, sim, 12)["available"]


def test_robot_pad_geometry_uses_measured_frame_and_rejects_invalid_or_nonbox_parts():
    rotation = Rotation.from_euler('xyz', [.3, .4, -.2]).as_matrix()
    position = np.array([1., 2., 3.])
    centers = np.array([[.02, 0, .01], [-.02, 0, .01]])
    gripper = SimpleNamespace(important_geoms={'left_fingerpad': ['left'], 'right_fingerpad': ['right']})
    robot = SimpleNamespace(gripper={'right': gripper}, eef_site_id={'right': 0})
    sim = SimpleNamespace(
        model=SimpleNamespace(geom_name2id={'left': 0, 'right': 1}.__getitem__, geom_type=[6, 6],
                              geom_size=np.array([[.008, .004, .008], [.008, .004, .008]])),
        data=SimpleNamespace(site_xpos=[position], site_xmat=[rotation.reshape(-1)],
                             geom_xpos=centers @ rotation.T + position,
                             geom_xmat=[rotation.reshape(-1)]*2))
    result = read_fingerpad_geometry(robot, sim, 17)
    assert result['available'] and result['step'] == 17 and result['frame'] == 'eef_control_site'
    np.testing.assert_allclose([p['center_eef'] for p in result['pads']], centers, atol=1e-12)
    np.testing.assert_allclose(result['pads'][0]['rotation_eef'], np.eye(3), atol=1e-12)
    sim.data.geom_xpos[0] += rotation[:, 0]*.01
    moved = read_fingerpad_geometry(robot, sim, 18)
    np.testing.assert_allclose(moved['pads'][0]['center_eef'], [.03, 0, .01], atol=1e-12)
    sim.model.geom_type[0] = 5
    assert not read_fingerpad_geometry(robot, sim, 18)['available']
    sim.model.geom_type[0] = 6
    sim.model.geom_size[0, 0] = np.nan
    assert not read_fingerpad_geometry(robot, sim, 18)['available']


async def test_feedback_target_uses_measured_site_not_policy_pose(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    # Deliberately corrupt policy pose; controller pose remains a separate measurement.
    h.obs.state[:3] = [10, 20, 30]
    control = OSCControl(h.obs)
    target = control.relative_target([0.04, 0, 0], [0, 0, 0.2], False)
    request = EEFRequest(target=target, steps=4)
    result = await execute_target(h, memory, lease, request, "tracking")
    np.testing.assert_allclose(h.env.pos, [0.04, 0, 0.1], atol=1e-12)
    assert result["position_error_m"] < 1e-12 and result["rotation_error_rad"] < 1e-12
    actions = [r["action"] for r in read_events(log.root) if r["kind"] == "action_executed"]
    np.testing.assert_allclose(np.array(actions)[:, 0], [1, 1, 0, 0], atol=1e-12)
    assert len(actions) == 4
    log.close()


async def test_controller_contract_is_required_and_stale_pose_rejected():
    obs = await MockEnvironment().observe()
    obs.eef_control_pose["step"] += 1
    with pytest.raises(ValueError, match="Stale"):
        OSCControl(obs)
    obs.eef_control_pose = {}
    with pytest.raises(ValueError, match="unavailable"):
        OSCControl(obs)


@pytest.mark.parametrize(
    "change",
    [
        {"position": [0.04, 0.04, 0.1]},
        {"base_motion": [0.1, 0, 0, 0]},
        {"control_mode": "base", "position": [0.001, 0, 0.1]},
        {"control_mode": "base", "base_motion": [0.26, 0, 0, 0]},
    ],
)
async def test_invalid_targets_rejected_before_any_execution(tmp_path, change):
    c, log, h, lease, memory = await make_harness(tmp_path)
    values = dict(position=[0, 0, 0.1], quaternion_xyzw=[0, 0, 0, 1], gripper_closed=False)
    values.update(change)
    with pytest.raises(ValueError):
        await execute_target(h, memory, lease, EEFRequest(target=EEFTarget(**values)), "bad")
    assert h.obs.step == 0
    log.close()


async def test_edit_matches_total_offset_and_noncommuting_rotation_composition():
    control = OSCControl(await MockEnvironment().observe())
    rng = np.random.default_rng(7)
    actions = rng.uniform(-0.05, 0.05, (5, 12))
    actions[:, [6, 11]] = rng.uniform(0, 1, (5, 2))
    dp, dr = np.array([0.005, -0.003, 0.002]), np.array([0.01, -0.02, 0.03])
    edited, diagnostics = control.edit(actions, ActionEdit(delta_position=dp, delta_rotation_vector=dr))
    original_rot, edited_rot = Rotation.identity(), Rotation.identity()
    original_pos, edited_pos = np.zeros(3), np.zeros(3)
    for i, (original, changed) in enumerate(zip(actions, edited)):
        old, new = control.scale(original[:6]), control.scale(changed[:6])
        original_pos += old[:3]
        edited_pos += new[:3]
        original_rot = Rotation.from_rotvec(old[3:]) * original_rot
        edited_rot = Rotation.from_rotvec(new[3:]) * edited_rot
        alpha = (i + 1) / len(actions)
        np.testing.assert_allclose(edited_pos, original_pos + alpha * dp, atol=1e-12)
        np.testing.assert_allclose(
            edited_rot.as_matrix(), (Rotation.from_rotvec(alpha * dr) * original_rot).as_matrix(), atol=1e-12
        )
    np.testing.assert_array_equal(actions[:, 6:], edited[:, 6:])
    assert not any(d["clipped"] for d in diagnostics)


async def test_edit_reports_clipping_and_keeps_mobile_channels():
    control = OSCControl(await MockEnvironment().observe())
    original = np.zeros((2, 12))
    original[:, 0] = 1
    original[:, 7] = 0.15
    edited, diagnostics = control.edit(
        original, ActionEdit(delta_position=[0.02, 0, 0], delta_rotation_vector=[0, 0, 0], gripper="closed")
    )
    assert all(d["clipped"] for d in diagnostics)
    np.testing.assert_array_equal(edited[:, 7:], original[:, 7:])
    assert np.all(edited[:, 6] == 1)


async def test_preview_does_not_execute_and_epoch_change_invalidates_edit(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    await h.claim(Source.ASTRA)
    tools = PlannerToolkit(h, memory, SkillLibrary(ROOT / "skills"), MockVLA())
    preview = await tools.call("preview_vla", {"steps": 2})
    assert h.obs.step == 0 and len(preview["actions"]) == 2
    with pytest.raises(ValueError, match="prefix"):
        tools.take_proposal(preview["proposal_id"], 3)
    await h.claim(Source.ASTRA)
    with pytest.raises(ValueError, match="stale"):
        tools.take_proposal(preview["proposal_id"], 1)
    fresh = await tools.call("preview_vla", {"steps": 1})
    assert fresh["proposal_id"] != preview["proposal_id"] and h.obs.step == 0
    log.close()


async def test_crop_is_recorded_rgb_and_future_reads_are_rejected(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    tools = PlannerToolkit(h, memory, SkillLibrary(ROOT / "skills"))
    result = await tools.call("view_image", dict(camera="left", step=0, bbox=[20, 20, 35, 35]))
    assert isinstance(result, ImageResult)
    crop = np.asarray(Image.open(io.BytesIO(base64.b64decode(result.url.split(",", 1)[1]))))
    np.testing.assert_array_equal(crop, h.obs.rgb["left"][20:35, 20:35])
    with pytest.raises(ValueError, match="causal"):
        await tools.call("view_image", dict(camera="left", step=1))
    with pytest.raises(ValueError):
        await tools.call("observe", dict(reset=True))
    log.close()


async def test_execution_read_distinguishes_command_from_measurement(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    action = np.zeros(12)
    action[0] = 0.5
    await h.execute(action, lease, memory)
    tools = PlannerToolkit(h, memory, SkillLibrary(ROOT / "skills"))
    rows = (await tools.call("read_execution", dict(start_step=0, end_step=1)))["actions"]
    assert rows[0]["action"][0] == 0.5 and rows[0]["measured_state"][0] == 0.01
    assert rows[0]["before"] != rows[0]["after"]
    log.close()


async def test_code_helper_counts_every_step_against_recovery_budget(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    c.runtime.recovery_steps = 2
    source = "def run(ctx):\n    ctx.move_relative([.04,0,0], False, steps=5)\n    return 'RECOVERED'"
    result = await RecoveryRunner(h, memory, c.runtime, log).run(source, "object_lifted", "helper-budget")
    assert result["result"] == "UNRESOLVED" and result["executed_steps"] == h.obs.step == 2
    with pytest.raises(StaleControl):
        await h.execute(np.zeros(12), lease, memory)
    log.close()


async def test_code_stop_does_not_execute_following_action(tmp_path):
    c, log, h, lease, memory = await make_harness(tmp_path)
    result = await RecoveryRunner(h, memory, c.runtime, log).run(
        "def run(ctx):\n    ctx.stop('need new observation')\n    ctx.act([0]*12)", "object_lifted", "stop"
    )
    assert result["result"] == "UNRESOLVED" and h.halted and h.obs.step == 0
    log.close()


def test_robocasa_reads_site_in_arm_origin_without_policy_body_pose():
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    controller = SimpleNamespace(
        name="OSC_POSE",
        input_type="delta",
        input_ref_frame="base",
        impedance_mode="fixed",
        input_min=-1,
        input_max=1,
        output_min=-0.05,
        output_max=0.05,
    )
    composite = SimpleNamespace(
        name="HYBRID_MOBILE_BASE",
        part_controllers=dict(right=controller, right_gripper=None, base=None, torso=None),
        get_controller_base_pose=lambda part: (np.array([1, 2, 3]), rotation),
    )
    robot = SimpleNamespace(
        robot_model=type("PandaOmron", (), {})(), composite_controller=composite, eef_site_id={"right": 0}
    )
    data = SimpleNamespace(site_xpos=np.array([[1, 3, 3]]), site_xmat=np.array([rotation.reshape(9)]))
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(env=SimpleNamespace(robots=[robot], sim=SimpleNamespace(data=data)))
    )
    contract, pose = read_robot_control(env, step=12)
    np.testing.assert_allclose(pose["position"], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(pose["quaternion_xyzw"], [0, 0, 0, 1], atol=1e-12)
    assert pose["step"] == 12 and contract["input_min"] == [-1] * 6
