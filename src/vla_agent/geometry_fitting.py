"""Numerical diagnostics only: fitting supplied points never verifies their physical correspondence."""

import numpy as np


def fit_geometry(points: list[list[float]], kind: str) -> dict:
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not 4 <= len(p) <= 128:
        raise ValueError('Supply 4..128 three-dimensional points in one metric frame')
    if not np.all(np.isfinite(p)) or np.max(np.abs(p)) > 100:
        raise ValueError('Points must be finite and within +/-100 meters')
    if kind not in {'plane', 'circle'}:
        raise ValueError('Unknown fit kind')
    centroid = np.mean(p, axis=0)
    centered = p-centroid
    _, values, axes = np.linalg.svd(centered, full_matrices=False)
    result = dict(kind=kind, executed=False, input_authority='caller_provided_numeric_points',
                  sample_count=len(p), singular_values_m=values.tolist(),
                  note='Numerical fit only, not validated object identity, calibration, correspondence, contact or success. '
                       'All points must use the same metric frame; circle points must track the same physical marker. '
                       'Inspect residuals using observation evidence; do not discard points just to pass a threshold.')
    if values[1] < 1e-6 or values[1] < values[0]*1e-6:
        return dict(**result, identifiable=False, reason='Points do not resolve two independent spatial directions')
    normal = axes[2]
    perpendicular = centered @ normal
    plane = dict(centroid_m=centroid.tolist(), normal_unoriented=normal.tolist(),
                 offset_m=float(-normal@centroid), rms_residual_m=float(np.sqrt(np.mean(perpendicular**2))),
                 max_residual_m=float(np.max(np.abs(perpendicular))),
                 residuals_m=perpendicular.tolist())
    if kind == 'plane':
        return dict(**result, identifiable=True, plane=plane)
    uv = centered @ axes[:2].T
    design = np.column_stack((2*uv, np.ones(len(p))))
    coefficients, _, rank, spectrum = np.linalg.lstsq(design, np.sum(uv**2, axis=1), rcond=None)
    if rank != 3:
        return dict(**result, identifiable=False, reason='Circle system is rank deficient', plane=plane)
    center_uv = coefficients[:2]
    squared = float(coefficients[2]+center_uv@center_uv)
    if squared <= 0 or not np.isfinite(squared):
        return dict(**result, identifiable=False, reason='No finite positive fitted radius', plane=plane)
    radius = float(np.sqrt(squared))
    center = centroid+center_uv@axes[:2]
    radial = np.linalg.norm(uv-center_uv, axis=1)-radius
    residual = np.sqrt(radial**2+perpendicular**2)
    angles = np.sort(np.mod(np.arctan2((uv-center_uv)[:,1], (uv-center_uv)[:,0]),2*np.pi))
    gaps = np.diff(np.r_[angles,angles[0]+2*np.pi])
    span = float(2*np.pi-np.max(gaps))
    return dict(**result, identifiable=True, plane=plane,
                circle=dict(center_m=center.tolist(), normal_unoriented=normal.tolist(), radius_m=radius,
                    rms_residual_m=float(np.sqrt(np.mean(residual**2))), max_residual_m=float(residual.max()),
                    residuals_m=residual.tolist(), radial_residuals_m=radial.tolist(), arc_span_rad=span,
                    design_condition_number=float(spectrum[0]/spectrum[-1]),
                    warning='A short arc can have low residual while its center/radius remain poorly constrained. '
                            'Fit quality alone does not establish that a hinge model applies.'))
