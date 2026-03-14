#!/usr/bin/env python3
"""Step 2 validator for COLMAP -> ACE pose and intrinsics conventions.

Checks:
1. Converted ACE camera centers match COLMAP formula C = -R^T t.
2. Converted ACE rotation matrices match inverse world-to-camera rotation.
3. Calibration file shape/value is consistent with COLMAP camera model path:
   - single-focal (1 value) for SIMPLE_* and RADIAL models
   - 3x3 matrix for PINHOLE/OPENCV-like models
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

COLMAP_CAMERA_MODELS = {
    "SIMPLE_PINHOLE": "single_focal",
    "SIMPLE_RADIAL": "single_focal",
    "RADIAL": "single_focal",
    "PINHOLE": "matrix",
    "OPENCV": "matrix",
    "OPENCV_FISHEYE": "matrix",
    "FULL_OPENCV": "matrix",
    "SIMPLE_RADIAL_FISHEYE": "single_focal",
    "RADIAL_FISHEYE": "single_focal",
}


@dataclass
class ColmapImage:
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str


@dataclass
class ColmapCamera:
    model: str
    width: int
    height: int
    params: list[float]


def parse_cameras_txt(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        cameras[camera_id] = ColmapCamera(
            model=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            params=[float(x) for x in parts[4:]],
        )
    return cameras


def parse_images_txt(path: Path) -> list[ColmapImage]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    images: list[ColmapImage] = []
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        images.append(
            ColmapImage(
                qw=float(parts[1]),
                qx=float(parts[2]),
                qy=float(parts[3]),
                qz=float(parts[4]),
                tx=float(parts[5]),
                ty=float(parts[6]),
                tz=float(parts[7]),
                camera_id=int(parts[8]),
                name=parts[9],
            )
        )
    return images


def expected_camera_center_and_rotation(img: ColmapImage) -> tuple[np.ndarray, np.ndarray]:
    rot = Rotation.from_quat([img.qx, img.qy, img.qz, img.qw]).as_matrix()
    t = np.array([img.tx, img.ty, img.tz], dtype=np.float64)
    center = -(rot.T @ t)
    return center, rot.T


def read_pose(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 4:
        raise ValueError(f"Pose file must have 4 lines: {path}")
    values: list[float] = []
    for line in lines:
        row = [float(x) for x in line.split()]
        if len(row) != 4:
            raise ValueError(f"Each pose line must have 4 values: {path}")
        values.extend(row)
    return np.array(values, dtype=np.float64).reshape(4, 4)


def read_calibration(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) == 1:
        return np.array([float(lines[0])], dtype=np.float64)
    rows: list[list[float]] = []
    for line in lines:
        rows.append([float(x) for x in line.split()])
    return np.array(rows, dtype=np.float64)


def rotation_error_deg(r_est: np.ndarray, r_ref: np.ndarray) -> float:
    delta = r_est @ r_ref.T
    trace = float(np.trace(delta))
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return float(np.degrees(np.arccos(cos_angle)))


def find_best_sparse_model(scene_dir: Path, ace_stems: set[str]) -> Path:
    sparse_root = scene_dir / "sparse"
    candidates: list[tuple[int, Path]] = []
    for subdir in sorted(sparse_root.iterdir()):
        images_txt = subdir / "images.txt"
        cameras_txt = subdir / "cameras.txt"
        if not (images_txt.exists() and cameras_txt.exists()):
            continue
        images = parse_images_txt(images_txt)
        stems = {Path(img.name).stem for img in images}
        matches = len(stems & ace_stems)
        candidates.append((matches, subdir))
    if not candidates:
        raise FileNotFoundError("No sparse submodel with text cameras/images found.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def validate(scene_dir: Path, ace_dir: Path, sparse_model_dir: Path | None) -> dict:
    ace_entries: list[tuple[str, str, Path, Path]] = []
    ace_stems: set[str] = set()

    for split in ("train", "test"):
        pose_dir = ace_dir / split / "poses"
        calib_dir = ace_dir / split / "calibration"
        for pose_file in sorted(pose_dir.glob("*.txt")):
            stem = pose_file.stem
            calib_file = calib_dir / f"{stem}.txt"
            if not calib_file.exists():
                raise FileNotFoundError(f"Missing calibration for {split}/{stem}")
            ace_entries.append((split, stem, pose_file, calib_file))
            ace_stems.add(stem)

    if sparse_model_dir is None:
        sparse_model_dir = find_best_sparse_model(scene_dir, ace_stems)

    cameras = parse_cameras_txt(sparse_model_dir / "cameras.txt")
    images = parse_images_txt(sparse_model_dir / "images.txt")
    image_by_stem = {Path(img.name).stem: img for img in images}

    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    focal_errors: list[float] = []
    matrix_errors: list[float] = []
    pose_bottom_row_errors: list[float] = []
    missing_stems: list[str] = []
    calibration_shape_mismatches: list[str] = []

    model_type_counts = {"single_focal": 0, "matrix": 0, "unknown": 0}

    for split, stem, pose_path, calib_path in ace_entries:
        img = image_by_stem.get(stem)
        if img is None:
            missing_stems.append(f"{split}/{stem}")
            continue

        cam = cameras[img.camera_id]
        model_type = COLMAP_CAMERA_MODELS.get(cam.model, "unknown")
        model_type_counts[model_type] = model_type_counts.get(model_type, 0) + 1

        pose = read_pose(pose_path)
        calib = read_calibration(calib_path)

        expected_center, expected_rotation = expected_camera_center_and_rotation(img)
        est_center = pose[:3, 3]
        est_rotation = pose[:3, :3]

        translation_errors.append(float(np.linalg.norm(est_center - expected_center)))
        rotation_errors.append(rotation_error_deg(est_rotation, expected_rotation))
        pose_bottom_row_errors.append(float(np.linalg.norm(pose[3, :] - np.array([0.0, 0.0, 0.0, 1.0]))))

        if model_type == "single_focal":
            if calib.ndim != 1 or calib.shape[0] != 1:
                calibration_shape_mismatches.append(f"{split}/{stem}: expected 1 value, got {calib.shape}")
            else:
                focal_errors.append(abs(float(calib[0]) - float(cam.params[0])))
        elif model_type == "matrix":
            expected_k = np.array(
                [
                    [cam.params[0], 0.0, cam.params[2]],
                    [0.0, cam.params[1], cam.params[3]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            if calib.shape != (3, 3):
                calibration_shape_mismatches.append(f"{split}/{stem}: expected 3x3, got {calib.shape}")
            else:
                matrix_errors.append(float(np.max(np.abs(calib - expected_k))))
        else:
            calibration_shape_mismatches.append(
                f"{split}/{stem}: unsupported camera model {cam.model}"
            )

    def summarize(values: list[float]) -> dict:
        if not values:
            return {"count": 0, "median": None, "max": None}
        arr = np.array(values, dtype=np.float64)
        return {
            "count": int(arr.size),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "scene_dir": str(scene_dir),
        "ace_dir": str(ace_dir),
        "sparse_model_dir": str(sparse_model_dir),
        "ace_entries": len(ace_entries),
        "colmap_images": len(images),
        "matched_entries": len(translation_errors),
        "missing_stems": missing_stems,
        "model_type_counts": model_type_counts,
        "translation_error_m": summarize(translation_errors),
        "rotation_error_deg": summarize(rotation_errors),
        "pose_bottom_row_error": summarize(pose_bottom_row_errors),
        "focal_error_px": summarize(focal_errors),
        "matrix_error_abs": summarize(matrix_errors),
        "calibration_shape_mismatches": calibration_shape_mismatches,
        "pass": {
            "all_ace_stems_found_in_colmap": len(missing_stems) == 0,
            "calibration_shapes_consistent": len(calibration_shape_mismatches) == 0,
            "translation_error_max_below_1e-6": (
                summarize(translation_errors)["max"] is not None
                and summarize(translation_errors)["max"] < 1e-6
            ),
            "rotation_error_max_below_1e-4_deg": (
                summarize(rotation_errors)["max"] is not None
                and summarize(rotation_errors)["max"] < 1e-4
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate COLMAP->ACE pose and intrinsics conventions.")
    parser.add_argument("--scene-dir", required=True, help="Scene root containing sparse/ and ace/.")
    parser.add_argument("--ace-dir", default=None, help="ACE directory (defaults to <scene-dir>/ace).")
    parser.add_argument(
        "--sparse-model-dir",
        default=None,
        help="Sparse model directory containing cameras.txt/images.txt. If omitted, auto-select by stem overlap.",
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    ace_dir = Path(args.ace_dir).resolve() if args.ace_dir else scene_dir / "ace"
    sparse_model_dir = Path(args.sparse_model_dir).resolve() if args.sparse_model_dir else None

    result = validate(scene_dir=scene_dir, ace_dir=ace_dir, sparse_model_dir=sparse_model_dir)

    output = json.dumps(result, indent=2)
    print(output)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
