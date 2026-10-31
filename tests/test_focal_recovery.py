"""Camera recovery must preserve the pixel coordinates of sampled points.

Reference: microsoft/MoGe@07444410, geometry_torch.recover_focal_shift:
nearest-resize both the point map and its original normalized_view_plane_uv.
The analytic fixture provides independently known focal length and shift.
"""
import math

import numpy as np
import pytest

from moge_mlx.model import _recover_focal_shift_np


@pytest.mark.parametrize('height,width', [(64, 64), (512, 512), (1254, 1254), (73, 121)])
@pytest.mark.parametrize('masked', [False, True])
def test_known_camera_survives_recovery_downsampling(height, width, masked):
    aspect = width / height
    diagonal = math.sqrt(1 + aspect * aspect)
    x, y = np.meshgrid(
        ((np.arange(width) + 0.5) / width * 2 - 1) * aspect / diagonal,
        ((np.arange(height) + 0.5) / height * 2 - 1) / diagonal,
    )
    focal, shift = 3.25, 0.7
    depth = 1 + 0.2 * np.sin(x * 8) * np.cos(y * 8)
    points = np.stack((x / focal * depth, y / focal * depth, depth - shift), axis=-1)
    mask = ((x + 0.13) ** 2 + y ** 2 < 0.35) if masked else np.ones_like(x, dtype=bool)
    actual_focal, actual_shift = _recover_focal_shift_np(
        points.astype(np.float32)[None], mask[None], height, width, aspect
    )
    assert actual_focal == pytest.approx(focal, rel=2e-5)
    assert actual_shift == pytest.approx(shift, abs=2e-5)
