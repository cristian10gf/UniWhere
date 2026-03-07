"""Pose loading, ACE results parsing, and calibration."""

import logging
import math
from pathlib import Path

import cv2
import numpy as np

_logger = logging.getLogger(__name__)


def load_split_poses(scene_dir: Path, split: str):
    """Load camera poses and image paths from an ACE dataset split."""
    split_dir = scene_dir / split
    pose_dir = split_dir / "poses"
    rgb_dir = split_dir / "rgb"

    if not pose_dir.exists():
        return [], []

    pose_files = sorted(pose_dir.iterdir())
    rgb_files = sorted(rgb_dir.iterdir())

    poses = [np.loadtxt(pf) for pf in pose_files]
    image_paths = [str(f) for f in rgb_files]
    return poses, image_paths


def parse_ace_results(results_file: Path):
    """
    Parse ACE test results file.
    Each line: filename qw qx qy qz tx ty tz rot_err trans_err inlier_count
    """
    results = []
    with open(results_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue

            filename = parts[0]
            qw, qx, qy, qz = (float(parts[i]) for i in range(1, 5))
            tx, ty, tz = (float(parts[i]) for i in range(5, 8))
            rot_err, trans_err = float(parts[8]), float(parts[9])
            inlier_count = int(parts[10]) if len(parts) > 10 else 0

            angle = 2 * math.acos(max(-1, min(1, qw)))
            if angle > 1e-6:
                axis = np.array([qx, qy, qz])
                norm = np.linalg.norm(axis)
                if norm > 1e-6:
                    axis /= norm
                R, _ = cv2.Rodrigues(axis * angle)
            else:
                R = np.eye(3)

            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = [tx, ty, tz]
            c2w = np.linalg.inv(w2c)

            results.append({
                "filename": filename,
                "pose_c2w": c2w,
                "rot_err": rot_err,
                "trans_err": trans_err,
                "inlier_count": inlier_count,
            })

    return results


def load_calibration(scene_dir: Path):
    """Load a representative calibration from the dataset."""
    for split in ("train", "test"):
        cal_dir = scene_dir / split / "calibration"
        if not cal_dir.exists():
            continue
        cal_files = sorted(cal_dir.iterdir())
        if not cal_files:
            continue
        cal = np.loadtxt(cal_files[0])
        if cal.ndim == 0:
            return float(cal)
        if cal.shape == (3, 3):
            return cal
        return float(cal.flat[0])
    return None
