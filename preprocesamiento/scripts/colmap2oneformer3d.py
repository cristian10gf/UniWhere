#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "numpy",
#     "open3d>=0.17",
#     "plyfile",
# ]
# ///
"""
Convert COLMAP reconstruction output to OneFormer3D S3DIS-compatible .npy format.

Supports both dense (fused.ply) and sparse (points3D.bin/txt) reconstructions.
Output is a .npy file with shape [N, 9]: x, y, z, r, g, b, nx, ny, nz
where RGB is normalized to [0, 1] (S3DIS convention).

Usage:
    uv run colmap2oneformer3d.py --colmap-dir data/serie-1 --output-dir data/serie-1/oneformer3d/input
    uv run colmap2oneformer3d.py --colmap-dir data/serie-1 --mode sparse --output-dir /tmp/test
    uv run colmap2oneformer3d.py --colmap-dir data/_merged --mode dense --voxel-size 0.03
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def find_dense_ply(colmap_dir: Path) -> Path | None:
    """Locate fused.ply inside a COLMAP dense reconstruction."""
    candidates = [
        colmap_dir / "dense" / "0" / "fused.ply",
        colmap_dir / "dense" / "fused.ply",
        colmap_dir / "fused.ply",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_sparse_dir(colmap_dir: Path) -> Path | None:
    """Locate a valid sparse reconstruction directory."""
    sparse_base = colmap_dir / "sparse"
    if not sparse_base.exists():
        return None

    for subdir in sorted(sparse_base.iterdir()):
        if not subdir.is_dir():
            continue
        if (subdir / "points3D.bin").exists() or (subdir / "points3D.txt").exists():
            return subdir
    return None


def load_dense_ply(ply_path: Path, voxel_size: float, estimate_normals: bool):
    """Load a dense fused.ply using open3d.

    Returns:
        coords: [N, 3] float64
        colors: [N, 3] float64 in [0, 1]
        normals: [N, 3] float64
    """
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"Dense PLY loaded: {len(pcd.points):,} points from {ply_path}")

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"After voxel downsampling ({voxel_size}): {len(pcd.points):,} points")

    coords = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)  # open3d reads as [0, 1]

    if not pcd.has_normals() or estimate_normals:
        print("Estimating normals (KNN=30, radius=0.1)...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
    normals = np.asarray(pcd.normals)

    return coords, colors, normals


def parse_points3d_bin(bin_path: Path):
    """Parse COLMAP points3D.bin (binary format).

    Binary format per point:
      point3D_id (uint64), x,y,z (float64×3), r,g,b (uint8×3),
      error (float64), track_length (uint64),
      then track_length × (image_id uint32, point2D_idx uint32)

    Returns:
        coords: [N, 3] float64
        colors: [N, 3] uint8
    """
    coords_list = []
    colors_list = []

    with open(bin_path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            _point3d_id = struct.unpack("<Q", f.read(8))[0]
            x, y, z = struct.unpack("<ddd", f.read(24))
            r, g, b = struct.unpack("<BBB", f.read(3))
            _error = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            # Skip track entries: each is (image_id u32 + point2d_idx u32) = 8 bytes
            f.read(track_len * 8)

            coords_list.append([x, y, z])
            colors_list.append([r, g, b])

    return np.array(coords_list, dtype=np.float64), np.array(colors_list, dtype=np.uint8)


def parse_points3d_txt(txt_path: Path):
    """Parse COLMAP points3D.txt (text format).

    Format: POINT3D_ID X Y Z R G B ERROR TRACK[] ...

    Returns:
        coords: [N, 3] float64
        colors: [N, 3] uint8
    """
    coords_list = []
    colors_list = []

    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            coords_list.append([x, y, z])
            colors_list.append([r, g, b])

    return np.array(coords_list, dtype=np.float64), np.array(colors_list, dtype=np.uint8)


def load_sparse_points(sparse_dir: Path, voxel_size: float):
    """Load a sparse point cloud from COLMAP and estimate normals.

    Returns:
        coords: [N, 3] float64
        colors: [N, 3] float64 in [0, 1]
        normals: [N, 3] float64
    """
    import open3d as o3d

    bin_path = sparse_dir / "points3D.bin"
    txt_path = sparse_dir / "points3D.txt"

    if bin_path.exists():
        coords, colors_u8 = parse_points3d_bin(bin_path)
        print(f"Sparse binary loaded: {len(coords):,} points from {bin_path}")
    elif txt_path.exists():
        coords, colors_u8 = parse_points3d_txt(txt_path)
        print(f"Sparse text loaded: {len(coords):,} points from {txt_path}")
    else:
        raise FileNotFoundError(f"No points3D.bin or points3D.txt in {sparse_dir}")

    colors = colors_u8.astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"After voxel downsampling ({voxel_size}): {len(pcd.points):,} points")

    print("Estimating normals (KNN=30, radius=0.1)...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    return np.asarray(pcd.points), np.asarray(pcd.colors), np.asarray(pcd.normals)


def main():
    parser = argparse.ArgumentParser(
        description="Convert COLMAP dense/sparse output to OneFormer3D S3DIS .npy format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--colmap-dir",
        required=True,
        help="COLMAP series directory (contains dense/ and/or sparse/).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory (scene.npy will be written here).",
    )
    parser.add_argument(
        "--mode",
        choices=["dense", "sparse", "auto"],
        default="auto",
        help="Which COLMAP output to use. 'auto' tries dense first, then sparse.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="Voxel size for downsampling (0 to disable).",
    )
    parser.add_argument(
        "--estimate-normals",
        action="store_true",
        default=False,
        help="Force re-estimation of normals even if the PLY already has them.",
    )
    args = parser.parse_args()

    colmap_dir = Path(args.colmap_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not colmap_dir.is_dir():
        print(f"Error: COLMAP directory not found: {colmap_dir}")
        sys.exit(1)

    # Detect mode
    dense_ply = find_dense_ply(colmap_dir)
    sparse_dir = find_sparse_dir(colmap_dir)

    if args.mode == "dense":
        if dense_ply is None:
            print(f"Error: no fused.ply found in {colmap_dir}")
            sys.exit(1)
        coords, colors, normals = load_dense_ply(
            dense_ply, args.voxel_size, args.estimate_normals
        )
    elif args.mode == "sparse":
        if sparse_dir is None:
            print(f"Error: no sparse reconstruction found in {colmap_dir}/sparse/")
            sys.exit(1)
        coords, colors, normals = load_sparse_points(sparse_dir, args.voxel_size)
    else:  # auto
        if dense_ply is not None:
            print(f"Auto-detected dense reconstruction: {dense_ply}")
            coords, colors, normals = load_dense_ply(
                dense_ply, args.voxel_size, args.estimate_normals
            )
        elif sparse_dir is not None:
            print(
                f"WARNING: dense reconstruction not found, falling back to sparse: {sparse_dir}"
            )
            print(
                "  For best results, re-run COLMAP with --dense flag to produce fused.ply"
            )
            coords, colors, normals = load_sparse_points(sparse_dir, args.voxel_size)
        else:
            print(f"Error: no COLMAP reconstruction found in {colmap_dir}")
            print("  Expected dense/0/fused.ply or sparse/0/points3D.bin")
            sys.exit(1)

    # Stack as [N, 9]: x, y, z, r, g, b, nx, ny, nz
    scene = np.column_stack([coords, colors, normals]).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scene.npy"
    np.save(output_path, scene)

    bbox_min = coords.min(axis=0).round(2)
    bbox_max = coords.max(axis=0).round(2)
    print(f"\nConversion complete:")
    print(f"  Points     : {len(scene):,}")
    print(f"  Shape      : {scene.shape}")
    print(f"  Bounding box: {bbox_min} → {bbox_max}")
    print(f"  Output     : {output_path}")


if __name__ == "__main__":
    main()
