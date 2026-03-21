"""Rerun logging/visualization and PLY export."""

import logging
from pathlib import Path

import numpy as np
import rerun as rr

_logger = logging.getLogger(__name__)


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


def log_to_rerun(
    pc_positions: np.ndarray,
    pc_colors: np.ndarray | None,
    mapping_poses: list,
    mapping_images: list,
    test_poses: list,
    test_images: list,
    ace_results: list | None,
    calibration,
    ace_pc_positions: np.ndarray | None = None,
    ace_pc_colors: np.ndarray | None = None,
):
    """Log all data to Rerun for interactive visualization."""

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # --- Original COLMAP point cloud ---
    if pc_positions is not None:
        if pc_colors is None:
            pc_colors = np.full((pc_positions.shape[0], 3), 200, dtype=np.uint8)

        rr.log(
            "world/colmap_point_cloud",
            rr.Points3D(positions=pc_positions, colors=pc_colors, radii=0.005),
            static=True,
        )
        _logger.info(f"Logged COLMAP point cloud: {pc_positions.shape[0]} points")

    # --- ACE-extracted point cloud (optional, separate entity) ---
    if ace_pc_positions is not None:
        if ace_pc_colors is None:
            ace_pc_colors = np.full((ace_pc_positions.shape[0], 3), 128, dtype=np.uint8)

        rr.log(
            "world/ace_point_cloud",
            rr.Points3D(positions=ace_pc_positions, colors=ace_pc_colors, radii=0.004),
            static=True,
        )
        _logger.info(f"Logged ACE network point cloud: {ace_pc_positions.shape[0]} points")

    pinhole = get_pinhole_params(calibration)

    # --- Mapping cameras (sparse - every Nth) ---
    skip = max(1, len(mapping_poses) // 50)
    logged_mapping = 0
    for i in range(0, len(mapping_poses), skip):
        pose = mapping_poses[i]
        entity = f"world/cameras/mapping/{i:05d}"
        rr.log(entity, rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]), static=True)
        if pinhole:
            fx, fy, cx, cy = pinhole
            rr.log(entity, rr.Pinhole(
                focal_length=[fx, fy], principal_point=[cx, cy],
                resolution=[int(cx * 2), int(cy * 2)], image_plane_distance=0.15,
            ), static=True)
        logged_mapping += 1
    _logger.info(f"Logged {logged_mapping}/{len(mapping_poses)} mapping cameras")

    # --- Test: estimated poses from ACE results ---
    if ace_results:
        est_positions = []
        est_colors = []

        for i, result in enumerate(ace_results):
            rr.set_time("frame", sequence=i)

            c2w = result["pose_c2w"]
            t = c2w[:3, 3]
            R = c2w[:3, :3]

            rr.log("world/cameras/estimated", rr.Transform3D(translation=t, mat3x3=R))

            if pinhole:
                fx, fy, cx, cy = pinhole
                rr.log("world/cameras/estimated", rr.Pinhole(
                    focal_length=[fx, fy], principal_point=[cx, cy],
                    resolution=[int(cx * 2), int(cy * 2)], image_plane_distance=0.4,
                ))

            # Find and log query image
            fname = Path(result["filename"]).name
            for tp in test_images:
                if Path(tp).name == fname:
                    if Path(tp).exists():
                        from skimage import io as skio
                        img = skio.imread(tp)
                        rr.log("world/cameras/estimated/image", rr.Image(img))
                        rr.log("query/image", rr.Image(img))
                    break

            rr.log("metrics/rotation_error_deg", rr.Scalars([float(result["rot_err"])]))
            rr.log(
                "metrics/translation_error_cm",
                rr.Scalars([float(result["trans_err"] * 100)]),
            )
            rr.log("metrics/inlier_count", rr.Scalars([float(result["inlier_count"])]))

            err_color = error_to_color(result["rot_err"], result["trans_err"])
            est_positions.append(t)
            est_colors.append(err_color)

        # Log all estimated positions as colored trail
        rr.log(
            "world/estimates_trail",
            rr.Points3D(positions=est_positions, colors=est_colors, radii=0.02),
            static=True,
        )
        _logger.info(f"Logged {len(ace_results)} ACE estimated poses")

    elif test_poses:
        for i, (pose, img_path) in enumerate(zip(test_poses, test_images)):
            rr.set_time("frame", sequence=i)
            rr.log("world/cameras/test", rr.Transform3D(
                translation=pose[:3, 3], mat3x3=pose[:3, :3]))

            if pinhole:
                fx, fy, cx, cy = pinhole
                rr.log("world/cameras/test", rr.Pinhole(
                    focal_length=[fx, fy], principal_point=[cx, cy],
                    resolution=[int(cx * 2), int(cy * 2)], image_plane_distance=0.4,
                ))

            if Path(img_path).exists():
                from skimage import io as skio
                img = skio.imread(img_path)
                rr.log("world/cameras/test/image", rr.Image(img))
                rr.log("query/image", rr.Image(img))

        _logger.info(f"Logged {len(test_poses)} GT test cameras")

    # GT test trajectory
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
    _logger.info(f"Exported PLY: {output_path}")
