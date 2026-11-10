import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_runtime import make_harness

from vla_agent.data import read_events
from vla_agent.geometry_fitting import fit_geometry
from vla_agent.memory import SkillLibrary
from vla_agent.toolkit import PlannerToolkit


def test_circle_recovers_metric_center_in_rotated_frame_and_reports_short_arc():
    rotation = Rotation.from_euler('xyz', [.3, -.7, .2]).as_matrix()
    center = np.array([.5, -.2, .7])
    angles = np.linspace(.2, 1.1, 12)
    points = np.column_stack((.15*np.cos(angles), .15*np.sin(angles), np.zeros(12)))
    points = points @ rotation.T + center
    result = fit_geometry(points.tolist(), 'circle')
    assert result['identifiable'] and not result['executed']
    assert result['input_authority'] == 'caller_provided_numeric_points'
    circle = result['circle']
    np.testing.assert_allclose(circle['center_m'], center, atol=1e-12)
    assert circle['radius_m'] == pytest.approx(.15)
    assert circle['arc_span_rad'] == pytest.approx(.9)
    assert abs(np.dot(circle['normal_unoriented'], rotation[:, 2])) == pytest.approx(1.)
    assert circle['max_residual_m'] < 1e-12
    assert 'short arc' in circle['warning']
    # An outlier remains in the fit and in its diagnostics; the tool must not silently drop it.
    points[0] += rotation[:, 2]*.03
    noisy = fit_geometry(points.tolist(), 'circle')
    assert len(noisy['circle']['residuals_m']) == 12
    assert noisy['circle']['max_residual_m'] > .01


def test_plane_has_no_spurious_success_or_physical_correspondence_claim():
    points = [[0., 0., .3], [.2, 0., .3], [0., .2, .3], [.2, .2, .3]]
    result = fit_geometry(points, 'plane')
    assert result['identifiable'] and result['plane']['rms_residual_m'] < 1e-12
    assert 'Numerical fit only' in result['note']
    assert 'success' not in result
    for kind in ('circle', 'plane'):
        degenerate = fit_geometry([[float(i), 0., .3] for i in range(4)], kind)
        assert not degenerate['identifiable']


@pytest.mark.parametrize('points', [[], [[0., 0., 0.]]*3, [[0., 0.]]*4,
                                  [[0., 0., float('nan')]]*4, [[101., 0., 0.]]*4])
def test_invalid_points_rejected(points):
    with pytest.raises(ValueError):
        fit_geometry(points, 'circle')


async def test_fit_available_in_readonly_review_without_control_or_execution(tmp_path):
    c, log, harness, lease, memory = await make_harness(tmp_path)
    toolkit = PlannerToolkit.for_review(harness, memory, SkillLibrary(c.runtime.skill_library),
                                       c.runtime.allowed_interventions)
    owner, epoch, step = harness.owner, harness.epoch, harness.obs.step
    result = await toolkit.call('fit_geometry', dict(kind='circle', points=[
        [.1, 0., 0.], [0., .1, 0.], [-.1, 0., 0.], [0., -.1, 0.]]))
    assert result['identifiable'] and result['circle']['radius_m'] == pytest.approx(.1)
    assert (harness.owner, harness.epoch, harness.obs.step) == (owner, epoch, step)
    assert not any(r['kind'] == 'action_executed' for r in read_events(log.root))
    log.close()
