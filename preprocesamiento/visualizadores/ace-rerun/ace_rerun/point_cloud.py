"""COLMAP point cloud loading and subsampling."""

import logging
import struct
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)


def load_ply(ply_path: Path):
    """
    Load a PLY point cloud file (binary or ASCII).
    Returns (positions Nx3, colors Nx3 uint8) or (positions, None).
    """
    from plyfile import PlyData

    _logger.info(f"Loading PLY: {ply_path}")
    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]

    x = np.array(vertex["x"], dtype=np.float32)
    y = np.array(vertex["y"], dtype=np.float32)
    z = np.array(vertex["z"], dtype=np.float32)
    positions = np.column_stack([x, y, z])

    colors = None
    try:
        r = np.array(vertex["red"], dtype=np.uint8)
        g = np.array(vertex["green"], dtype=np.uint8)
        b = np.array(vertex["blue"], dtype=np.uint8)
        colors = np.column_stack([r, g, b])
    except ValueError:
        pass

    _logger.info(f"Loaded {positions.shape[0]} points from PLY")
    return positions, colors


def load_colmap_points3d_txt(path: Path):
    """Parse COLMAP points3D.txt: POINT3D_ID X Y Z R G B ERROR TRACK[]."""
    positions = []
    colors = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            positions.append([x, y, z])
            colors.append([r, g, b])

    positions = np.array(positions, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)
    _logger.info(f"Loaded {positions.shape[0]} points from points3D.txt")
    return positions, colors


def load_colmap_points3d_bin(path: Path):
    """Parse COLMAP points3D.bin (binary format)."""
    positions = []
    colors = []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point3d_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<ddd", f.read(24))
            rgb = struct.unpack("<BBB", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_length = struct.unpack("<Q", f.read(8))[0]
            f.read(track_length * 8)  # skip track entries (image_id + point2d_idx)
            positions.append(xyz)
            colors.append(rgb)

    positions = np.array(positions, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)
    _logger.info(f"Loaded {positions.shape[0]} points from points3D.bin")
    return positions, colors


def find_colmap_point_cloud(colmap_dir: Path):
    """
    Auto-detect the best point cloud from a COLMAP directory.
    Priority: dense/0/fused.ply > sparse/0/points3D.txt > sparse/0/points3D.bin
    """
    candidates = [
        colmap_dir / "dense" / "0" / "fused.ply",
        colmap_dir / "dense" / "fused.ply",
    ]
    for c in candidates:
        if c.exists():
            return load_ply(c)

    sparse_dirs = []
    sparse_root = colmap_dir / "sparse"
    if sparse_root.exists():
        for sub in sorted(sparse_root.iterdir()):
            if sub.is_dir():
                sparse_dirs.append(sub)
        if not sparse_dirs:
            sparse_dirs = [sparse_root]

    for sd in sparse_dirs:
        txt = sd / "points3D.txt"
        if txt.exists():
            return load_colmap_points3d_txt(txt)
        binf = sd / "points3D.bin"
        if binf.exists():
            return load_colmap_points3d_bin(binf)

    raise FileNotFoundError(
        f"No point cloud found in {colmap_dir}. "
        "Expected dense/0/fused.ply or sparse/0/points3D.{{txt,bin}}"
    )


def subsample_point_cloud(positions, colors, max_points):
    """Randomly subsample a point cloud if it exceeds max_points."""
    n = positions.shape[0]
    if n <= max_points:
        return positions, colors
    indices = np.random.choice(n, max_points, replace=False)
    indices.sort()
    new_colors = colors[indices] if colors is not None else None
    _logger.info(f"Subsampled point cloud: {n} -> {max_points} points")
    return positions[indices], new_colors
