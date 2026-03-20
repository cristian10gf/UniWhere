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


# ---------------------------------------------------------------------------
# Writer tests (Task 2)
# ---------------------------------------------------------------------------
import tempfile
from mast3r_sfm import write_cameras_txt, write_images_txt, write_points3d_txt, write_ply, deduplicate_sparse_pts3d


def test_write_cameras_txt_shared():
    focals = [800.0, 800.0]
    pps = [[960.0, 540.0], [960.0, 540.0]]
    shapes = [(1080, 1920), (1080, 1920)]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cameras.txt"
        write_cameras_txt(p, focals, pps, shapes, shared_intrinsics=True)
        lines = [l for l in p.read_text().splitlines() if not l.startswith('#') and l.strip()]
        assert len(lines) == 1
        parts = lines[0].split()
        assert parts[0] == '1'
        assert parts[1] == 'PINHOLE'
        assert parts[2] == '1920'   # width
        assert parts[3] == '1080'   # height
        assert abs(float(parts[4]) - 800.0) < 0.01   # fx
        assert abs(float(parts[6]) - 960.0) < 0.01   # cx
        assert abs(float(parts[7]) - 540.0) < 0.01   # cy


def test_write_cameras_txt_per_image():
    focals = [800.0, 820.0]
    pps = [[960.0, 540.0], [970.0, 545.0]]
    shapes = [(1080, 1920), (1080, 1920)]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cameras.txt"
        write_cameras_txt(p, focals, pps, shapes, shared_intrinsics=False)
        lines = [l for l in p.read_text().splitlines() if not l.startswith('#') and l.strip()]
        assert len(lines) == 2
        assert lines[0].split()[0] == '1'
        assert lines[1].split()[0] == '2'


def test_write_images_txt_identity_pose():
    c2w = np.eye(4)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "images.txt"
        write_images_txt(p, [c2w], ["frame000000.jpg"], shared_intrinsics=True)
        lines = [l for l in p.read_text().splitlines()
                 if not l.startswith('#') and l.strip()]
        assert len(lines) == 1  # pose line only (empty points line stripped by l.strip())
        parts = lines[0].split()
        assert parts[0] == '1'
        assert abs(float(parts[1]) - 1.0) < 1e-6   # qw
        assert abs(float(parts[5]) - 0.0) < 1e-6   # tx
        assert parts[9] == 'frame000000.jpg'


def test_write_points3d_txt_filters_nan():
    pts = [np.array([[float('nan'), 0, 0], [1.0, 2.0, 3.0]])]
    cols = [np.array([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]])]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "points3D.txt"
        n = write_points3d_txt(p, pts, cols)
        assert n == 1   # solo el punto finito
        data = [l for l in p.read_text().splitlines() if not l.startswith('#') and l.strip()]
        assert len(data) == 1


def test_deduplicate_sparse_pts3d_removes_duplicates():
    """Puntos idénticos (mismo voxel) deben colapsar a uno solo."""
    rng = np.random.default_rng(0)
    # 5 puntos únicos reales
    unique_pts = rng.random((5, 3)) * 10.0
    cols_base  = rng.random((5, 3))
    # Cada punto duplicado 3 veces con ruido sub-voxel (10× menor que el voxel)
    # voxel_size = diagonal * 0.002; con diagonal ~sqrt(3)*10≈17, voxel≈0.034
    # ruido de 0.001 << 0.034 → todos caen en el mismo voxel
    noise = rng.random((5, 3, 3)) * 0.001
    pts_list  = [unique_pts + noise[:, i, :] for i in range(3)]
    cols_list = [cols_base + rng.random((5, 3)) * 0.001 for _ in range(3)]
    pts_out, cols_out = deduplicate_sparse_pts3d(pts_list, cols_list)
    # 5 puntos únicos × 3 copias → después de dedup deben quedar exactamente 5
    assert pts_out.shape == (5, 3)
    assert cols_out.shape == (5, 3)


def test_deduplicate_sparse_pts3d_preserves_geometry():
    """Puntos bien separados deben mantenerse casi todos (pérdida ≤ 1% por filtro outlier)."""
    # 100 puntos en una cuadrícula 10×10, separados 1.0 en X e Y
    grid = np.array([[i, j, 0.0] for i in range(10) for j in range(10)])
    cols = np.zeros((100, 3))
    pts_out, _ = deduplicate_sparse_pts3d([grid], [cols])
    # Con diagonal ≈ 12.7 y voxel_size = 0.002*12.7 ≈ 0.025,
    # puntos separados 1.0 >> 0.025 → casi todos sobreviven.
    # El filtro outlier 99.9 percentil puede descartar el punto de mayor norma.
    assert len(pts_out) >= 99


def test_deduplicate_sparse_pts3d_filters_nan():
    """NaN e Inf deben eliminarse antes de la deduplicación."""
    pts = [np.array([[float('nan'), 0, 0], [1.0, 0.0, 0.0], [float('inf'), 0, 0]])]
    cols = [np.zeros((3, 3))]
    pts_out, _ = deduplicate_sparse_pts3d(pts, cols)
    assert len(pts_out) == 1  # solo el punto finito


def test_write_ply_confidence_filter():
    pts = [np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])]  # shape (1, 2, 3)
    confs = [np.array([[0.5, 2.0]])]   # primer punto bajo umbral, segundo pasa
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fused.ply"
        n = write_ply(p, pts, confs, colors_list=None, min_conf_thr=1.5)
        assert n == 1
        content = p.read_text()
        assert "element vertex 1" in content
