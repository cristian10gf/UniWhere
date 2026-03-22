"""
Relocalización de imágenes query usando el modelo ACE entrenado.

Módulo autónomo — no depende del paquete backend/.
Reutiliza el mecanismo setup_ace_imports de ace_extraction.py para
importar Regressor desde el directorio fuente de ACE.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

_logger = logging.getLogger(__name__)

_IMAGE_MEAN = 0.4
_IMAGE_STD = 0.25
_DEFAULT_IMAGE_HEIGHT = 480
_OUTPUT_SUBSAMPLE = 8


# ---------------------------------------------------------------------------
# PnP solver (duplicado de backend/relocalization/pose_solver.py)
# ---------------------------------------------------------------------------

@dataclass
class PnPResult:
    """Resultado de estimación de pose PnP."""
    pose_4x4: np.ndarray
    inlier_count: int
    success: bool


def solve_pose_pnp(
    scene_coordinates_3HW: torch.Tensor,
    focal_length: float,
    pp_x: float,
    pp_y: float,
    reprojection_threshold: float = 10.0,
    iterations: int = 64,
    output_subsample: int = _OUTPUT_SUBSAMPLE,
) -> PnPResult:
    """
    Estima pose de cámara desde coordenadas de escena ACE via OpenCV PnP+RANSAC.

    Parámetros
    ----------
    scene_coordinates_3HW : torch.Tensor
        Coordenadas predichas por la red ACE, shape (3, H_out, W_out).
    focal_length : float
        Focal length en píxeles (ya escalado a la resolución de entrada).
    pp_x, pp_y : float
        Punto principal (ya escalado).
    reprojection_threshold : float
        Umbral de inliers RANSAC en píxeles.
    iterations : int
        Iteraciones RANSAC.
    output_subsample : int
        Factor de submuestreo del output de la red (default: 8).
    """
    sc = scene_coordinates_3HW.detach().cpu().float().numpy()
    _, h_out, w_out = sc.shape

    half = output_subsample / 2.0
    xs = np.arange(w_out, dtype=np.float64) * output_subsample + half
    ys = np.arange(h_out, dtype=np.float64) * output_subsample + half
    grid_x, grid_y = np.meshgrid(xs, ys)

    points_2d = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    points_3d = sc.reshape(3, -1).T.astype(np.float64)

    valid = np.isfinite(points_3d).all(axis=1) & (np.abs(points_3d).sum(axis=1) > 1e-6)
    points_2d = points_2d[valid]
    points_3d = points_3d[valid]

    if len(points_2d) < 4:
        return PnPResult(pose_4x4=np.eye(4), inlier_count=0, success=False)

    camera_matrix = np.array([
        [focal_length, 0.0, pp_x],
        [0.0, focal_length, pp_y],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=points_3d,
        imagePoints=points_2d,
        cameraMatrix=camera_matrix,
        distCoeffs=None,
        iterationsCount=iterations,
        reprojectionError=reprojection_threshold,
        flags=cv2.SOLVEPNP_P3P,
    )

    if not ok or inliers is None:
        return PnPResult(pose_4x4=np.eye(4), inlier_count=0, success=False)

    R, _ = cv2.Rodrigues(rvec)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = tvec.ravel()
    c2w = np.linalg.inv(w2c)

    return PnPResult(pose_4x4=c2w, inlier_count=len(inliers), success=True)


# ---------------------------------------------------------------------------
# Query result y relocalizador
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Resultado de relocalización de una imagen query."""
    image_path: Path
    pose_c2w: np.ndarray | None   # 4×4 camera-to-world, None si success=False
    inlier_count: int
    success: bool


def _setup_ace_imports(ace_dir: Path) -> None:
    """Añade el directorio ACE a sys.path para importar Regressor."""
    ace_dir = ace_dir.resolve()
    if not ace_dir.exists():
        raise FileNotFoundError(f"Directorio ACE no encontrado: {ace_dir}")
    if str(ace_dir) not in sys.path:
        sys.path.insert(0, str(ace_dir))


class QueryRelocalizer:
    """
    Carga un modelo ACE entrenado y estima poses 6DoF para imágenes query.

    Parámetros
    ----------
    encoder_path : Path
        Path al encoder pre-entrenado (ace_encoder_pretrained.pt).
    head_path : Path
        Path al head de la escena (.pt).
    ace_dir : Path
        Directorio fuente de ACE (para importar Regressor).
    image_height : int
        Altura de imagen para inferencia (default: 480).
    """

    def __init__(
        self,
        encoder_path: Path,
        head_path: Path,
        ace_dir: Path,
        image_height: int = _DEFAULT_IMAGE_HEIGHT,
    ):
        self.image_height = image_height

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._use_half = device.type == "cuda"

        _setup_ace_imports(ace_dir)
        from ace_network import Regressor  # importado desde ace_dir  # noqa: PLC0415

        encoder_state = torch.load(encoder_path, map_location="cpu", weights_only=True)
        head_state = torch.load(head_path, map_location="cpu", weights_only=True)

        self._network = Regressor.create_from_split_state_dict(encoder_state, head_state)
        self._network = self._network.to(device).eval()

        if self._use_half:
            self._network = self._network.half()

        _logger.info(
            f"QueryRelocalizer listo en {device} "
            f"(half={self._use_half}, height={image_height})"
        )

    def _preprocess(self, image_path: Path, focal_length: float):
        """Carga imagen, redimensiona y normaliza para ACE. Retorna (tensor, scaled_K)."""
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"No se pudo cargar imagen: {image_path}")

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        orig_h, orig_w = gray.shape

        scale = self.image_height / orig_h
        new_w = int(orig_w * scale)
        resized = cv2.resize(gray, (new_w, self.image_height), interpolation=cv2.INTER_AREA)

        scaled_focal = focal_length * scale
        scaled_pp_x = (orig_w / 2.0) * scale
        scaled_pp_y = (orig_h / 2.0) * scale

        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = (tensor - _IMAGE_MEAN) / _IMAGE_STD
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        if self._use_half:
            tensor = tensor.half()

        return tensor, scaled_focal, scaled_pp_x, scaled_pp_y

    def relocalize_images(
        self,
        image_paths: list[Path],
        focal_length: float,
        reprojection_threshold: float = 10.0,
        ransac_iterations: int = 64,
    ) -> list[QueryResult]:
        """
        Estima la pose de cámara para cada imagen query.

        Parámetros
        ----------
        image_paths : list[Path]
            Paths a las imágenes query.
        focal_length : float
            Focal length en píxeles a la resolución original.
        reprojection_threshold : float
            Umbral RANSAC en píxeles (default: 10.0).
        ransac_iterations : int
            Iteraciones RANSAC (default: 64).

        Retorna
        -------
        list[QueryResult]
            Un resultado por imagen, en el mismo orden.
        """
        from torch.amp import autocast

        results = []
        for img_path in image_paths:
            _logger.info(f"Relocalizando: {img_path.name}")
            try:
                tensor, sf, sx, sy = self._preprocess(img_path, focal_length)
                tensor = tensor.to(self.device, non_blocking=True)

                with torch.no_grad():
                    with autocast(self.device.type, enabled=self._use_half):
                        scene_coords = self._network(tensor)

                sc_3hw = scene_coords[0].float().cpu()
                pnp = solve_pose_pnp(
                    scene_coordinates_3HW=sc_3hw,
                    focal_length=sf,
                    pp_x=sx,
                    pp_y=sy,
                    reprojection_threshold=reprojection_threshold,
                    iterations=ransac_iterations,
                )

                results.append(QueryResult(
                    image_path=img_path,
                    pose_c2w=pnp.pose_4x4 if pnp.success else None,
                    inlier_count=pnp.inlier_count,
                    success=pnp.success,
                ))

            except Exception as exc:
                _logger.error(f"Error relocalizando '{img_path.name}': {exc}")
                results.append(QueryResult(
                    image_path=img_path,
                    pose_c2w=None,
                    inlier_count=0,
                    success=False,
                ))

        return results
