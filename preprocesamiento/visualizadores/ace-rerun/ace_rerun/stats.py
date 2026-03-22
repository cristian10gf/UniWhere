"""Estadísticas del modelo ACE y del dataset."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)


@dataclass
class ModelStats:
    """Estadísticas estructurales y de accuracy del modelo ACE."""
    # Info estructural
    n_train_images: int
    n_test_images: int
    model_size_mb: float | None          # None si no hay modelo .pt
    image_resolution: tuple | None       # (w, h) de la primera imagen de train

    # Métricas de test (None si no hay ace_results)
    pct_5cm_5deg: float | None
    pct_10cm_10deg: float | None
    pct_50cm_5deg: float | None
    median_trans_err_cm: float | None    # en centímetros
    median_rot_err_deg: float | None     # en grados
    mean_inliers: float | None


def _count_images(split_dir: Path) -> int:
    """Cuenta archivos de imagen en <split>/rgb/."""
    rgb_dir = split_dir / "rgb"
    if not rgb_dir.exists():
        return 0
    return len([f for f in rgb_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def _get_image_resolution(split_dir: Path) -> tuple | None:
    """Lee resolución de la primera imagen de train (sin cargarla completa)."""
    try:
        import cv2
        rgb_dir = split_dir / "rgb"
        if not rgb_dir.exists():
            return None
        images = sorted(rgb_dir.iterdir())
        if not images:
            return None
        img = cv2.imread(str(images[0]))
        if img is None:
            return None
        h, w = img.shape[:2]
        return (w, h)
    except Exception:
        return None


def compute_stats(
    scene_dir: Path,
    model_path: Path | None,
    ace_results: list | None,
) -> ModelStats:
    """
    Calcula estadísticas del modelo ACE y del dataset.

    Parámetros
    ----------
    scene_dir : Path
        Directorio ACE con train/ y test/.
    model_path : Path | None
        Path al archivo .pt del head entrenado (para obtener tamaño).
    ace_results : list | None
        Resultados de parse_ace_results() — lista de dicts con
        'rot_err' (grados), 'trans_err' (metros), 'inlier_count'.
    """
    train_dir = scene_dir / "train"
    test_dir = scene_dir / "test"

    n_train = _count_images(train_dir)
    n_test = _count_images(test_dir)
    resolution = _get_image_resolution(train_dir)

    model_size_mb = None
    if model_path is not None and model_path.exists():
        model_size_mb = model_path.stat().st_size / (1024 * 1024)

    # Sin resultados de test
    if not ace_results:
        return ModelStats(
            n_train_images=n_train,
            n_test_images=n_test,
            model_size_mb=model_size_mb,
            image_resolution=resolution,
            pct_5cm_5deg=None,
            pct_10cm_10deg=None,
            pct_50cm_5deg=None,
            median_trans_err_cm=None,
            median_rot_err_deg=None,
            mean_inliers=None,
        )

    trans_errs_m = np.array([r["trans_err"] for r in ace_results], dtype=float)
    rot_errs_deg = np.array([r["rot_err"] for r in ace_results], dtype=float)
    inliers = np.array([r["inlier_count"] for r in ace_results], dtype=float)
    n = len(ace_results)

    def _pct(t_thresh_m, r_thresh_deg):
        mask = (trans_errs_m < t_thresh_m) & (rot_errs_deg < r_thresh_deg)
        return float(mask.sum()) / n * 100.0

    return ModelStats(
        n_train_images=n_train,
        n_test_images=n_test,
        model_size_mb=model_size_mb,
        image_resolution=resolution,
        pct_5cm_5deg=_pct(0.05, 5.0),
        pct_10cm_10deg=_pct(0.10, 10.0),
        pct_50cm_5deg=_pct(0.50, 5.0),
        median_trans_err_cm=float(np.median(trans_errs_m) * 100),
        median_rot_err_deg=float(np.median(rot_errs_deg)),
        mean_inliers=float(np.mean(inliers)),
    )
