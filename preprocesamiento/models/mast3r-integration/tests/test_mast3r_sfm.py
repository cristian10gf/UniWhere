import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from mast3r_sfm import rot_to_quat_wxyz, c2w_to_colmap_pose


def quat_to_rot(w, x, y, z) -> np.ndarray:
    """Reconstruye R desde cuaternión (numpy-only, sin scipy)."""
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)    ],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z),  2*(y*z - x*w)    ],
        [2*(x*z - y*w),     2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ])


def test_rot_to_quat_identity():
    w, x, y, z = rot_to_quat_wxyz(np.eye(3))
    assert abs(w - 1.0) < 1e-6
    assert abs(x) < 1e-6 and abs(y) < 1e-6 and abs(z) < 1e-6


def test_rot_to_quat_180_around_z():
    R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
    w, x, y, z = rot_to_quat_wxyz(R)
    R_back = quat_to_rot(w, x, y, z)
    assert np.allclose(R, R_back, atol=1e-5)


def test_rot_to_quat_roundtrip_arbitrary():
    # Rotar 45° alrededor de eje (1,1,1)/sqrt(3)
    theta = np.pi / 4
    ax = np.array([1, 1, 1]) / np.sqrt(3)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    w, x, y, z = rot_to_quat_wxyz(R)
    R_back = quat_to_rot(w, x, y, z)
    assert np.allclose(R, R_back, atol=1e-5)


def test_c2w_to_colmap_pose_identity():
    qw, qx, qy, qz, tx, ty, tz = c2w_to_colmap_pose(np.eye(4))
    assert abs(qw - 1.0) < 1e-6
    assert abs(tx) < 1e-6 and abs(ty) < 1e-6 and abs(tz) < 1e-6


def test_c2w_to_colmap_pose_translation():
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    _, _, _, _, tx, ty, tz = c2w_to_colmap_pose(c2w)
    assert abs(tx - (-1.0)) < 1e-6
    assert abs(ty - (-2.0)) < 1e-6
    assert abs(tz - (-3.0)) < 1e-6
