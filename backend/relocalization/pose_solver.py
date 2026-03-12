"""
OpenCV PnP+RANSAC pose solver.

Replaces dsacstar.forward_rgb() for inference-only relocalization.
Converts ACE scene coordinate predictions into a 6DoF camera pose
using cv2.solvePnPRansac().
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass
class PnPResult:
    """Result of PnP pose estimation."""
    pose_4x4: np.ndarray        # 4×4 camera-to-world matrix
    inlier_count: int           # number of RANSAC inliers
    success: bool               # whether solvePnPRansac succeeded


# Subsample factor of the ACE network output relative to the input image.
_OUTPUT_SUBSAMPLE = 8


def solve_pose(
    scene_coordinates_3HW: torch.Tensor,
    focal_length: float,
    pp_x: float,
    pp_y: float,
    reprojection_threshold: float = 10.0,
    iterations: int = 64,
    output_subsample: int = _OUTPUT_SUBSAMPLE,
) -> PnPResult:
    """
    Estimate a camera pose from predicted scene coordinates via OpenCV PnP+RANSAC.

    Parameters
    ----------
    scene_coordinates_3HW : torch.Tensor
        Scene coordinates predicted by the ACE network, shape (3, H_out, W_out).
        H_out = H_in / output_subsample, W_out = W_in / output_subsample.
    focal_length : float
        Focal length in pixels (already scaled to the network input resolution).
    pp_x, pp_y : float
        Principal point in pixels (already scaled to the network input resolution).
    reprojection_threshold : float
        RANSAC inlier threshold in pixels.
    iterations : int
        Number of RANSAC iterations.
    output_subsample : int
        Subsample factor of the network output (default: 8).

    Returns
    -------
    PnPResult
        Contains the 4×4 camera-to-world pose, inlier count, and success flag.
    """
    sc = scene_coordinates_3HW.detach().cpu().float().numpy()  # (3, H, W)
    _, h_out, w_out = sc.shape

    # Build 2D pixel grid matching the subsampled output.
    # Each output pixel corresponds to the center of an 8×8 patch in the input.
    half = output_subsample / 2.0
    xs = np.arange(w_out, dtype=np.float64) * output_subsample + half
    ys = np.arange(h_out, dtype=np.float64) * output_subsample + half
    grid_x, grid_y = np.meshgrid(xs, ys)

    # Flatten to (N, 2) for 2D points and (N, 3) for 3D points.
    points_2d = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)  # (N, 2)
    points_3d = sc.reshape(3, -1).T.astype(np.float64)               # (N, 3)

    # Filter out invalid scene coordinates (NaN, inf, or all-zero).
    valid = np.isfinite(points_3d).all(axis=1) & (np.abs(points_3d).sum(axis=1) > 1e-6)
    points_2d = points_2d[valid]
    points_3d = points_3d[valid]

    if len(points_2d) < 4:
        return PnPResult(
            pose_4x4=np.eye(4),
            inlier_count=0,
            success=False,
        )

    # Camera intrinsics matrix.
    camera_matrix = np.array([
        [focal_length, 0.0, pp_x],
        [0.0, focal_length, pp_y],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=points_3d,
        imagePoints=points_2d,
        cameraMatrix=camera_matrix,
        distCoeffs=None,
        iterationsCount=iterations,
        reprojectionError=reprojection_threshold,
        flags=cv2.SOLVEPNP_P3P,
    )

    if not success or inliers is None:
        return PnPResult(
            pose_4x4=np.eye(4),
            inlier_count=0,
            success=False,
        )

    # Convert rotation vector to rotation matrix.
    R, _ = cv2.Rodrigues(rvec)

    # Build world-to-camera 4×4 matrix, then invert to get camera-to-world.
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = tvec.ravel()

    c2w = np.linalg.inv(w2c)

    return PnPResult(
        pose_4x4=c2w,
        inlier_count=len(inliers),
        success=True,
    )
