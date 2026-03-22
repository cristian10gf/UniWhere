import numpy as np
import pytest
from pathlib import Path
from ace_rerun.stats import compute_stats, ModelStats


def _make_ace_results(n: int, trans_err: float = 0.05, rot_err: float = 2.0):
    """Crea lista de resultados ACE sintéticos."""
    results = []
    for i in range(n):
        results.append({
            "filename": f"frame_{i:05d}.jpg",
            "pose_c2w": np.eye(4),
            "rot_err": rot_err,
            "trans_err": trans_err,   # en metros
            "inlier_count": 100,
        })
    return results


def test_compute_stats_no_model_no_results(tmp_path):
    # Crear estructura mínima de escena ACE
    (tmp_path / "train" / "rgb").mkdir(parents=True)
    for i in range(5):
        (tmp_path / "train" / "rgb" / f"img_{i}.jpg").touch()
    (tmp_path / "test" / "rgb").mkdir(parents=True)
    for i in range(3):
        (tmp_path / "test" / "rgb" / f"img_{i}.jpg").touch()

    stats = compute_stats(scene_dir=tmp_path, model_path=None, ace_results=None)

    assert stats.n_train_images == 5
    assert stats.n_test_images == 3
    assert stats.model_size_mb is None
    assert stats.pct_5cm_5deg is None


def test_compute_stats_with_results(tmp_path):
    (tmp_path / "train" / "rgb").mkdir(parents=True)
    (tmp_path / "test" / "rgb").mkdir(parents=True)

    # 10 resultados: todos dentro de 5cm/5°
    results = _make_ace_results(10, trans_err=0.03, rot_err=1.0)
    stats = compute_stats(scene_dir=tmp_path, model_path=None, ace_results=results)

    assert stats.pct_5cm_5deg == pytest.approx(100.0)
    assert stats.median_trans_err_cm == pytest.approx(3.0)
    assert stats.median_rot_err_deg == pytest.approx(1.0)
    assert stats.mean_inliers == pytest.approx(100.0)


def test_compute_stats_pct_thresholds(tmp_path):
    (tmp_path / "train" / "rgb").mkdir(parents=True)
    (tmp_path / "test" / "rgb").mkdir(parents=True)

    # 4 buenos (0.03m/1°) + 6 malos (1.0m/20°)
    good = _make_ace_results(4, trans_err=0.03, rot_err=1.0)
    bad = _make_ace_results(6, trans_err=1.0, rot_err=20.0)
    results = good + bad

    stats = compute_stats(scene_dir=tmp_path, model_path=None, ace_results=results)

    assert stats.pct_5cm_5deg == pytest.approx(40.0)   # 4/10


def test_compute_stats_model_size(tmp_path):
    (tmp_path / "train" / "rgb").mkdir(parents=True)
    (tmp_path / "test" / "rgb").mkdir(parents=True)
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"0" * 1024 * 1024 * 4)  # 4 MB

    stats = compute_stats(scene_dir=tmp_path, model_path=model_path, ace_results=None)

    assert stats.model_size_mb == pytest.approx(4.0, rel=0.01)
