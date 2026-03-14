#!/usr/bin/env python3
"""Step 1 audit: COLMAP sparse vs ACE dataset parity for one scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_registered_image_names(images_txt: Path) -> list[str]:
    lines = [
        line.strip()
        for line in images_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    names: list[str] = []
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) >= 10:
            names.append(parts[9])
    return names


def calibration_format(calib_file: Path) -> str:
    rows = [row.split() for row in calib_file.read_text(encoding="utf-8", errors="ignore").splitlines() if row.strip()]
    if len(rows) == 1 and len(rows[0]) == 1:
        return "single_value"
    if len(rows) == 3 and all(len(r) == 3 for r in rows):
        return "matrix_3x3"
    return "invalid"


def split_report(ace_dir: Path, split: str) -> tuple[dict, list[str]]:
    split_dir = ace_dir / split
    rgb_files = sorted(p for p in (split_dir / "rgb").iterdir() if p.is_file())
    pose_files = sorted(p for p in (split_dir / "poses").iterdir() if p.is_file())
    cal_files = sorted(p for p in (split_dir / "calibration").iterdir() if p.is_file())

    rgb_bases = [p.stem for p in rgb_files]
    pose_bases = [p.stem for p in pose_files]
    cal_bases = [p.stem for p in cal_files]

    rgb_set = set(rgb_bases)
    pose_set = set(pose_bases)
    cal_set = set(cal_bases)

    format_counts = {"single_value": 0, "matrix_3x3": 0, "invalid": 0}
    invalid_files: list[str] = []
    for cal_file in cal_files:
        fmt = calibration_format(cal_file)
        format_counts[fmt] += 1
        if fmt == "invalid" and len(invalid_files) < 5:
            invalid_files.append(cal_file.name)

    report = {
        "rgb_count": len(rgb_bases),
        "pose_count": len(pose_bases),
        "calibration_count": len(cal_bases),
        "missing_pose_for_rgb": len(rgb_set - pose_set),
        "missing_calibration_for_rgb": len(rgb_set - cal_set),
        "extra_pose_without_rgb": len(pose_set - rgb_set),
        "extra_calibration_without_rgb": len(cal_set - rgb_set),
        "sample_missing_pose": sorted(rgb_set - pose_set)[:5],
        "sample_missing_calibration": sorted(rgb_set - cal_set)[:5],
        "sample_extra_pose": sorted(pose_set - rgb_set)[:5],
        "sample_extra_calibration": sorted(cal_set - rgb_set)[:5],
        "calibration_formats": format_counts,
        "sample_invalid_calibration_files": invalid_files,
    }
    return report, rgb_bases


def parse_camera_models(cameras_txt: Path) -> dict[str, int]:
    models: dict[str, int] = {}
    for line in cameras_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        model = s.split()[1]
        models[model] = models.get(model, 0) + 1
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one scene for COLMAP-to-ACE step-1 parity.")
    parser.add_argument("--scene-dir", required=True, help="Path to scene directory containing sparse/, images/, ace/")
    args = parser.parse_args()

    scene = Path(args.scene_dir)
    images_txt = scene / "sparse" / "0" / "images.txt"
    cameras_txt = scene / "sparse" / "0" / "cameras.txt"
    images_dir = scene / "images"
    ace_dir = scene / "ace"

    registered_names = parse_registered_image_names(images_txt)
    registered_stems = {Path(name).stem for name in registered_names}

    train_report, train_bases = split_report(ace_dir, "train")
    test_report, test_bases = split_report(ace_dir, "test")

    ace_stems = set(train_bases + test_bases)
    missing_registered_stems = sorted(registered_stems - ace_stems)
    extra_ace_stems = sorted(ace_stems - registered_stems)

    report = {
        "scene": scene.name,
        "colmap_registered_images": len(registered_names),
        "colmap_unique_registered_stems": len(registered_stems),
        "images_present_on_disk": sum(1 for p in images_dir.rglob("*") if p.is_file()),
        "camera_models": parse_camera_models(cameras_txt),
        "ace_train": train_report,
        "ace_test": test_report,
        "ace_total_rgb": train_report["rgb_count"] + test_report["rgb_count"],
        "ace_total_unique_stems": len(ace_stems),
        "missing_registered_stems_in_ace": len(missing_registered_stems),
        "sample_missing_registered_stems_in_ace": missing_registered_stems[:10],
        "ace_stems_not_in_registered": len(extra_ace_stems),
        "sample_ace_stems_not_in_registered": extra_ace_stems[:10],
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
