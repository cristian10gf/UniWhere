"""Rerun logging/visualization and PLY export."""

import logging
from pathlib import Path

import cv2
import numpy as np
import rerun as rr

_logger = logging.getLogger(__name__)

_MAX_IMG_WIDTH = 640


def _resize_for_rerun(img: np.ndarray, max_width: int = _MAX_IMG_WIDTH) -> np.ndarray:
    """Redimensiona imagen a max_width px de ancho máximo (no upscalea)."""
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_pinhole_params(calibration):
    """Extract fx, fy, cx, cy from calibration."""
    if calibration is None:
        return None
    if isinstance(calibration, np.ndarray) and calibration.shape == (3, 3):
        return calibration[0, 0], calibration[1, 1], calibration[0, 2], calibration[1, 2]
    f = float(calibration)
    return f, f, 320.0, 240.0


def error_to_color(rot_err: float, trans_err: float,
                   thresh_cm: float = 20.0, thresh_deg: float = 5.0):
    """Map pose error to RGB: green=good, red=bad."""
    err = min(max(trans_err * 100 / thresh_cm, rot_err / thresh_deg), 1.0)
    return [int(255 * err), int(255 * (1 - err)), 0]


def _build_image_cache(image_paths: list) -> dict:
    """Carga y redimensiona todas las imágenes en memoria (más rápido que skimage)."""
    cache = {}
    for path in image_paths:
        p = Path(path)
        if not p.exists():
            continue
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        cache[p.name] = _resize_for_rerun(img_rgb)
    _logger.info(f"Image cache: {len(cache)}/{len(image_paths)} imágenes cargadas")
    return cache


def log_to_rerun(
    pc_positions: np.ndarray,
    pc_colors: np.ndarray | None,
    mapping_poses: list,
    test_poses: list,
    test_images: list,
    ace_results: list | None,
    calibration,
    ace_pc_positions: np.ndarray | None = None,
    ace_pc_colors: np.ndarray | None = None,
):
    """Log all data to Rerun for interactive visualization."""

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # --- Nube de puntos COLMAP ---
    if pc_positions is not None:
        if pc_colors is None:
            pc_colors = np.full((pc_positions.shape[0], 3), 200, dtype=np.uint8)
        rr.log(
            "world/colmap_point_cloud",
            rr.Points3D(positions=pc_positions, colors=pc_colors, radii=0.005),
            static=True,
        )
        _logger.info(f"Nube COLMAP: {pc_positions.shape[0]} puntos")

    # --- Nube de puntos ACE (opcional) ---
    if ace_pc_positions is not None:
        if ace_pc_colors is None:
            ace_pc_colors = np.full((ace_pc_positions.shape[0], 3), 128, dtype=np.uint8)
        rr.log(
            "world/ace_point_cloud",
            rr.Points3D(positions=ace_pc_positions, colors=ace_pc_colors, radii=0.004),
            static=True,
        )
        _logger.info(f"Nube ACE red: {ace_pc_positions.shape[0]} puntos")

    pinhole = get_pinhole_params(calibration)

    # --- Cámaras de mapping: batch (2 llamadas en vez de 2N) ---
    if mapping_poses:
        skip = max(1, len(mapping_poses) // 200)
        sampled = mapping_poses[::skip][:200]

        positions = np.array([p[:3, 3] for p in sampled], dtype=np.float32)
        # Vector de vista: tercera columna de R (dirección Z de la cámara en world)
        directions = np.array([p[:3, 2] for p in sampled], dtype=np.float32)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-9

        rr.log(
            "world/cameras/mapping",
            rr.Points3D(positions=positions,
                        colors=[[180, 180, 255]] * len(sampled),
                        radii=0.02),
            static=True,
        )
        rr.log(
            "world/cameras/mapping/view_dirs",
            rr.Arrows3D(origins=positions, vectors=directions * 0.15,
                        colors=[[100, 100, 200]] * len(sampled)),
            static=True,
        )
        _logger.info(f"Cámaras mapping: {len(sampled)}/{len(mapping_poses)} loggadas")

    # --- Pre-cargar imágenes de test en cache ---
    # Incluir tanto test_images como filenames de ace_results (pueden diferir)
    all_test_images = list(test_images) if test_images else []
    if ace_results:
        ace_image_paths = [r["filename"] for r in ace_results if Path(r["filename"]).exists()]
        all_test_images = list({Path(p).name: p for p in all_test_images + ace_image_paths}.values())
    image_cache = _build_image_cache(all_test_images)

    # --- Poses estimadas por ACE con timeline ---
    if ace_results:
        est_positions = []
        est_colors = []

        for i, result in enumerate(ace_results):
            rr.set_time("frame", sequence=i)

            c2w = result["pose_c2w"]
            t = c2w[:3, 3]
            R = c2w[:3, :3]

            rr.log("world/cameras/estimated",
                   rr.Transform3D(translation=t,
                                  mat3x3=rr.datatypes.Mat3x3(R)))

            if pinhole:
                fx, fy, cx, cy = pinhole
                rr.log("world/cameras/estimated", rr.Pinhole(
                    focal_length=[fx, fy], principal_point=[cx, cy],
                    resolution=[int(cx * 2), int(cy * 2)],
                    image_plane_distance=0.4,
                ))

            fname = Path(result["filename"]).name
            if fname in image_cache:
                img = image_cache[fname]
                rr.log("world/cameras/estimated/image", rr.Image(img))
                rr.log("query/image", rr.Image(img))

            rr.log("metrics/rotation_error_deg",
                   rr.Scalars(float(result["rot_err"])))
            rr.log("metrics/translation_error_cm",
                   rr.Scalars(float(result["trans_err"] * 100)))
            rr.log("metrics/inlier_count",
                   rr.Scalars(float(result["inlier_count"])))

            err_color = error_to_color(result["rot_err"], result["trans_err"])
            est_positions.append(t)
            est_colors.append(err_color)

        rr.log(
            "world/estimates_trail",
            rr.Points3D(positions=est_positions, colors=est_colors, radii=0.02),
            static=True,
        )
        _logger.info(f"Poses estimadas ACE: {len(ace_results)}")

    elif test_poses:
        for i, (pose, img_path) in enumerate(zip(test_poses, test_images)):
            rr.set_time("frame", sequence=i)
            t = pose[:3, 3]
            R = pose[:3, :3]
            rr.log("world/cameras/test",
                   rr.Transform3D(translation=t,
                                  mat3x3=rr.datatypes.Mat3x3(R)))

            if pinhole:
                fx, fy, cx, cy = pinhole
                rr.log("world/cameras/test", rr.Pinhole(
                    focal_length=[fx, fy], principal_point=[cx, cy],
                    resolution=[int(cx * 2), int(cy * 2)],
                    image_plane_distance=0.4,
                ))

            fname = Path(img_path).name
            if fname in image_cache:
                img = image_cache[fname]
                rr.log("world/cameras/test/image", rr.Image(img))
                rr.log("query/image", rr.Image(img))

        _logger.info(f"Cámaras GT test: {len(test_poses)}")

    # --- Trayectoria GT ---
    if test_poses:
        gt_positions = np.array([p[:3, 3] for p in test_poses])
        rr.log(
            "world/gt_trajectory",
            rr.Points3D(
                positions=gt_positions,
                colors=[[100, 100, 255]] * len(gt_positions),
                radii=0.01,
            ),
            static=True,
        )


def log_query_results(query_results: list, calibration) -> None:
    """
    Loggea resultados de relocalización de imágenes query como static.

    Parámetros
    ----------
    query_results : list[QueryResult]
        Lista de resultados de QueryRelocalizer.relocalize_images().
    calibration : np.ndarray | float | None
        Calibración de la cámara del dataset ACE.
    """
    pinhole = get_pinhole_params(calibration)
    success_positions = []

    for result in query_results:
        stem = result.image_path.stem
        entity = f"world/cameras/query/{stem}"

        if result.success and result.pose_c2w is not None:
            t = result.pose_c2w[:3, 3]
            R = result.pose_c2w[:3, :3]

            rr.log(entity,
                   rr.Transform3D(translation=t,
                                  mat3x3=rr.datatypes.Mat3x3(R)),
                   static=True)

            if pinhole:
                fx, fy, cx, cy = pinhole
                rr.log(entity, rr.Pinhole(
                    focal_length=[fx, fy], principal_point=[cx, cy],
                    resolution=[int(cx * 2), int(cy * 2)],
                    image_plane_distance=0.4,
                ), static=True)

            # Cargar y loggar imagen query
            img_bgr = cv2.imread(str(result.image_path))
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_rgb = _resize_for_rerun(img_rgb)
                rr.log(f"{entity}/image", rr.Image(img_rgb), static=True)

            success_positions.append(t)
            _logger.info(
                f"Query '{stem}': OK — inliers={result.inlier_count}, "
                f"t=[{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]"
            )
        else:
            rr.log(entity,
                   rr.TextLog(f"Relocalización fallida para '{stem}'",
                               level=rr.TextLogLevel.WARN),
                   static=True)
            _logger.warning(f"Query '{stem}': FAILED")

    if success_positions:
        rr.log(
            "world/query_trail",
            rr.Points3D(
                positions=success_positions,
                colors=[[255, 200, 0]] * len(success_positions),
                radii=0.04,
            ),
            static=True,
        )
    _logger.info(
        f"Query images: {len(success_positions)}/{len(query_results)} relocalizadas"
    )


def log_model_stats(stats) -> None:
    """
    Loggea estadísticas del modelo ACE a Rerun como TextDocument.

    Parámetros
    ----------
    stats : ModelStats
        Resultado de ace_rerun.stats.compute_stats().
    """
    model_size_str = (
        f"{stats.model_size_mb:.1f} MB" if stats.model_size_mb is not None
        else "N/A (sin modelo)"
    )
    res_str = (
        f"{stats.image_resolution[0]}×{stats.image_resolution[1]}"
        if stats.image_resolution else "desconocida"
    )

    model_md = f"""## Información del Modelo ACE

| Campo | Valor |
|-------|-------|
| Imágenes de entrenamiento | {stats.n_train_images} |
| Imágenes de test | {stats.n_test_images} |
| Tamaño del modelo | {model_size_str} |
| Resolución de imagen | {res_str} |
"""
    rr.log("stats/model_info",
           rr.TextDocument(model_md, media_type="text/markdown"),
           static=True)

    if stats.pct_5cm_5deg is not None:
        acc_md = f"""## Métricas de Accuracy ACE

| Umbral | % frames |
|--------|----------|
| 5 cm / 5° | {stats.pct_5cm_5deg:.1f}% |
| 10 cm / 10° | {stats.pct_10cm_10deg:.1f}% |
| 50 cm / 5° | {stats.pct_50cm_5deg:.1f}% |

| Estadística | Valor |
|-------------|-------|
| Mediana error traslación | {stats.median_trans_err_cm:.1f} cm |
| Mediana error rotación | {stats.median_rot_err_deg:.2f}° |
| Media de inliers | {stats.mean_inliers:.0f} |
"""
        rr.log("stats/accuracy",
               rr.TextDocument(acc_md, media_type="text/markdown"),
               static=True)


def export_ply(pc_xyz: np.ndarray, pc_clr: np.ndarray, output_path: Path):
    """Export point cloud to PLY."""
    from plyfile import PlyData, PlyElement

    if pc_clr is None:
        pc_clr = np.full((pc_xyz.shape[0], 3), 200, dtype=np.uint8)

    vertices = np.zeros(
        pc_xyz.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertices["x"] = pc_xyz[:, 0]
    vertices["y"] = pc_xyz[:, 1]
    vertices["z"] = pc_xyz[:, 2]
    vertices["red"] = pc_clr[:, 0]
    vertices["green"] = pc_clr[:, 1]
    vertices["blue"] = pc_clr[:, 2]

    PlyData([PlyElement.describe(vertices, "vertex")], text=True).write(str(output_path))
    _logger.info(f"PLY exportado: {output_path}")
