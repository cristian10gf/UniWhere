#!/usr/bin/env python3
"""Run MASt3R matching + mapper and adapt outputs to UniWhere's COLMAP layout."""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

import numpy as np
import pycolmap  # type: ignore[import-not-found]

import mast3r.utils.path_to_dust3r  # type: ignore[import-not-found]  # noqa: F401
from kapture.converter.colmap.database import COLMAPDatabase  # type: ignore[import-not-found]
from kapture.converter.colmap.database_extra import kapture_to_colmap  # type: ignore[import-not-found]
from mast3r.colmap.mapping import (  # type: ignore[import-not-found]
    glomap_run_mapper,
    kapture_import_image_folder_or_list,
    pycolmap_run_mapper,
    run_mast3r_matching,
)
from mast3r.model import AsymmetricMASt3R  # type: ignore[import-not-found]


VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MASt3R reconstruction bridge for UniWhere.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--series-dir", required=True, help="Series root directory.")
    parser.add_argument("--images-dir", required=True, help="Series images directory.")
    parser.add_argument(
        "--work-dir",
        default="",
        help="Intermediate MASt3R workspace (defaults to <series-dir>/mast3r).",
    )
    parser.add_argument(
        "--matcher",
        choices=["sequential", "exhaustive", "vocab_tree"],
        default="exhaustive",
        help="Pair generation strategy.",
    )
    parser.add_argument("--overlap", type=int, default=20, help="Sequential overlap window.")

    parser.add_argument("--model-name", default="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    parser.add_argument("--weights", default="", help="Optional local checkpoint path.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")

    parser.add_argument("--conf-thr", type=float, default=1.001)
    parser.add_argument("--pixel-tol", type=int, default=0)
    parser.add_argument("--dense-matching", action="store_true", default=False)
    parser.add_argument("--skip-geometric-verification", action="store_true", default=False)
    parser.add_argument("--min-len-track", type=int, default=5)

    parser.add_argument("--use-glomap-mapper", action="store_true", default=False)
    parser.add_argument("--glomap-bin", default="glomap")

    return parser.parse_args()


def list_relative_images(images_dir: Path) -> list[str]:
    relpaths: list[str] = []
    for file_path in sorted(images_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue
        relpaths.append(file_path.relative_to(images_dir).as_posix())
    return relpaths


def build_pairs(images: list[str], matcher: str, overlap: int) -> list[tuple[str, str]]:
    if len(images) < 2:
        return []

    if matcher == "sequential":
        window = max(1, overlap)
        pairs: list[tuple[str, str]] = []
        for i in range(len(images)):
            upper = min(len(images), i + window + 1)
            for j in range(i + 1, upper):
                pairs.append((images[i], images[j]))
        return pairs

    if matcher == "vocab_tree":
        print("[WARN] matcher=vocab_tree no aplica en MASt3R; usando exhaustive.")

    return list(itertools.combinations(images, 2))


def write_pairs_file(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for img1, img2 in pairs:
            f.write(f"{img1} {img2}\n")


def count_reg_images(reconstruction: pycolmap.Reconstruction) -> int:
    attr = getattr(reconstruction, "num_reg_images", None)
    if callable(attr):
        return int(attr())
    if isinstance(attr, int):
        return attr
    return len(getattr(reconstruction, "images", {}))


def count_points3d(reconstruction: pycolmap.Reconstruction) -> int:
    attr = getattr(reconstruction, "num_points3D", None)
    if callable(attr):
        return int(attr())
    if isinstance(attr, int):
        return attr
    return len(getattr(reconstruction, "points3D", {}))


def find_best_model(reconstruction_root: Path) -> tuple[Path, pycolmap.Reconstruction]:
    candidates = [d for d in sorted(reconstruction_root.iterdir()) if d.is_dir()]
    if not candidates:
        candidates = [reconstruction_root]

    best_path: Path | None = None
    best_reconstruction: pycolmap.Reconstruction | None = None
    best_score = (-1, -1)

    for model_dir in candidates:
        try:
            reconstruction = pycolmap.Reconstruction(str(model_dir))
        except Exception:
            continue

        score = (count_reg_images(reconstruction), count_points3d(reconstruction))
        if score > best_score:
            best_score = score
            best_path = model_dir
            best_reconstruction = reconstruction

    if best_path is None or best_reconstruction is None:
        raise RuntimeError(f"No valid reconstruction found in {reconstruction_root}")

    print(
        f"Selected model: {best_path.name} "
        f"({best_score[0]} registered images, {best_score[1]} points3D)."
    )
    return best_path, best_reconstruction


def write_ascii_ply(reconstruction: pycolmap.Reconstruction, ply_path: Path) -> int:
    points_data = []
    for point in getattr(reconstruction, "points3D", {}).values():
        xyz = np.asarray(getattr(point, "xyz", []), dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            continue

        color = np.asarray(getattr(point, "color", [255, 255, 255]), dtype=float)
        if color.shape != (3,):
            color = np.array([255, 255, 255], dtype=float)
        color = np.clip(np.round(color), 0, 255).astype(int)

        points_data.append((xyz, color))

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    with ply_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_data)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for xyz, color in points_data:
            f.write(
                f"{xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )

    return len(points_data)


def adapt_outputs(series_dir: Path, work_dir: Path, best_model_dir: Path, reconstruction: pycolmap.Reconstruction) -> None:
    db_src = work_dir / "colmap.db"
    db_dst = series_dir / "database.db"
    if not db_src.exists():
        raise FileNotFoundError(f"Expected database not found: {db_src}")

    sparse_dst = series_dir / "sparse" / "0"
    dense_dst = series_dir / "dense" / "0"

    if sparse_dst.exists():
        shutil.rmtree(sparse_dst)
    if dense_dst.exists():
        shutil.rmtree(dense_dst)

    sparse_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(best_model_dir, sparse_dst)
    shutil.copy2(db_src, db_dst)

    dense_dst.mkdir(parents=True, exist_ok=True)
    points_written = write_ascii_ply(reconstruction, dense_dst / "fused.ply")

    print(f"Adapted outputs into: {series_dir}")
    print(f"  database.db  : {db_dst}")
    print(f"  sparse model : {sparse_dst}")
    print(f"  dense cloud  : {dense_dst / 'fused.ply'} ({points_written} points)")


def main() -> int:
    args = parse_args()

    series_dir = Path(args.series_dir).resolve()
    images_dir = Path(args.images_dir).resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    work_dir = Path(args.work_dir).resolve() if args.work_dir else (series_dir / "mast3r")
    work_dir.mkdir(parents=True, exist_ok=True)

    image_relpaths = list_relative_images(images_dir)
    if len(image_relpaths) < 2:
        raise RuntimeError("Need at least 2 valid images to reconstruct.")

    pairs = build_pairs(image_relpaths, args.matcher, args.overlap)
    if not pairs:
        raise RuntimeError("No image pairs generated for matching.")

    pairs_file = work_dir / "pairs.txt"
    write_pairs_file(pairs_file, pairs)
    print(f"Generated {len(pairs)} image pairs -> {pairs_file}")

    weights_path = args.weights or f"naver/{args.model_name}"
    print(f"Loading model: {weights_path}")
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    maxdim = max(model.patch_embed.img_size)
    patch_size = model.patch_embed.patch_size

    kdata = kapture_import_image_folder_or_list(str(images_dir), use_single_camera=True)

    colmap_db_path = work_dir / "colmap.db"
    if colmap_db_path.exists():
        colmap_db_path.unlink()

    reconstruction_root = work_dir / "reconstruction"
    if reconstruction_root.exists():
        shutil.rmtree(reconstruction_root)
    reconstruction_root.mkdir(parents=True, exist_ok=True)

    colmap_db = COLMAPDatabase.connect(str(colmap_db_path))
    try:
        kapture_to_colmap(
            kdata,
            str(images_dir),
            tar_handler=None,
            database=colmap_db,
            keypoints_type=None,
            descriptors_type=None,
            export_two_view_geometry=False,
        )

        kept_pairs = run_mast3r_matching(
            model,
            maxdim,
            patch_size,
            args.device,
            kdata,
            str(images_dir),
            pairs,
            colmap_db,
            args.dense_matching,
            args.pixel_tol,
            args.conf_thr,
            args.skip_geometric_verification,
            args.min_len_track,
        )
    finally:
        colmap_db.close()

    if not kept_pairs:
        raise RuntimeError("No valid MASt3R matches were kept after filtering.")

    if not args.skip_geometric_verification:
        verify_pairs_file = work_dir / "pairs-verified.txt"
        write_pairs_file(verify_pairs_file, kept_pairs)
        pycolmap.verify_matches(str(colmap_db_path), str(verify_pairs_file))
        print(f"verify_matches completed with {len(kept_pairs)} pairs.")

    if args.use_glomap_mapper:
        glomap_run_mapper(args.glomap_bin, str(colmap_db_path), str(reconstruction_root), str(images_dir))
    else:
        pycolmap_run_mapper(str(colmap_db_path), str(reconstruction_root), str(images_dir))

    best_model_dir, best_reconstruction = find_best_model(reconstruction_root)
    adapt_outputs(series_dir, work_dir, best_model_dir, best_reconstruction)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
