#!/usr/bin/env python3
"""
Convert COLMAP sparse reconstruction output to ACE dataset format.

COLMAP stores poses as world-to-camera (quaternion + translation).
ACE expects camera-to-world 4x4 matrices and focal length or 3x3 intrinsics.

Usage:
    python colmap2ace.py --colmap-dir data/serie-1 --output-dir data/serie-1/ace
    python colmap2ace.py --colmap-dir data/serie-1 --output-dir data/serie-1/ace --train-ratio 0.9
"""

import argparse
import os
import random
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


COLMAP_CAMERA_MODELS = {
    'SIMPLE_PINHOLE': {'params': ['f', 'cx', 'cy'], 'type': 'single_focal'},
    'SIMPLE_RADIAL': {'params': ['f', 'cx', 'cy', 'k'], 'type': 'single_focal'},
    'RADIAL': {'params': ['f', 'cx', 'cy', 'k1', 'k2'], 'type': 'single_focal'},
    'PINHOLE': {'params': ['fx', 'fy', 'cx', 'cy'], 'type': 'matrix'},
    'OPENCV': {'params': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'], 'type': 'matrix'},
    'OPENCV_FISHEYE': {'params': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'], 'type': 'matrix'},
    'FULL_OPENCV': {
        'params': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'],
        'type': 'matrix',
    },
    'SIMPLE_RADIAL_FISHEYE': {'params': ['f', 'cx', 'cy', 'k'], 'type': 'single_focal'},
    'RADIAL_FISHEYE': {'params': ['f', 'cx', 'cy', 'k1', 'k2'], 'type': 'single_focal'},
}


def parse_cameras_txt(path):
    """Parse COLMAP cameras.txt, return dict camera_id -> {model, width, height, params}."""
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params,
            }
    return cameras


def parse_images_txt(path):
    """Parse COLMAP images.txt, return list of dicts with pose and image info.

    images.txt alternates: odd lines have image metadata, even lines have 2D points (skipped).
    Format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    """
    images = []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = parts[9]

        images.append({
            'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
            'tx': tx, 'ty': ty, 'tz': tz,
            'camera_id': camera_id,
            'name': name,
        })

    return images


def colmap_to_cam2world(qw, qx, qy, qz, tx, ty, tz):
    """Convert COLMAP world-to-camera pose to camera-to-world 4x4 matrix.

    COLMAP convention: R * X_world + t = X_camera
    ACE expects the inverse: X_world = R^T * (X_camera - t)
    """
    rot = Rotation.from_quat([qx, qy, qz, qw])
    R = rot.as_matrix()

    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = [tx, ty, tz]

    c2w = np.linalg.inv(w2c)
    return c2w


def extract_calibration(camera):
    """Extract calibration from COLMAP camera model.

    Returns (value, type) where:
      - type='focal': value is a single float (focal length)
      - type='matrix': value is a 3x3 numpy intrinsics matrix
    """
    model = camera['model']
    params = camera['params']

    if model not in COLMAP_CAMERA_MODELS:
        raise ValueError(f"Unsupported COLMAP camera model: {model}")

    model_info = COLMAP_CAMERA_MODELS[model]

    if model_info['type'] == 'single_focal':
        return params[0], 'focal'
    else:
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ])
        return K, 'matrix'


def write_pose(path, pose_4x4):
    """Write 4x4 pose matrix to text file (ACE format)."""
    with open(path, 'w') as f:
        for row in range(4):
            values = ' '.join(str(float(pose_4x4[row, col])) for col in range(4))
            f.write(values + '\n')


def write_calibration(path, calib_value, calib_type):
    """Write calibration file: single float or 3x3 matrix."""
    with open(path, 'w') as f:
        if calib_type == 'focal':
            f.write(str(float(calib_value)))
        else:
            for row in range(3):
                values = ' '.join(str(float(calib_value[row, col])) for col in range(3))
                f.write(values + '\n')


def _read_bin_count(path):
    """Read the leading uint64 entry count from a COLMAP binary model file."""
    with open(path, 'rb') as f:
        return int(struct.unpack('<Q', f.read(8))[0])


def _count_text_entries(path, kind):
    """Count entries in COLMAP text model files."""
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if kind == 'images':
        return len(lines) // 2
    return len(lines)


def _get_sparse_model_stats(model_dir):
    """Collect comparable stats for one sparse submodel directory."""
    cameras_txt = model_dir / 'cameras.txt'
    images_txt = model_dir / 'images.txt'
    points_txt = model_dir / 'points3D.txt'

    cameras_bin = model_dir / 'cameras.bin'
    images_bin = model_dir / 'images.bin'
    points_bin = model_dir / 'points3D.bin'

    has_camera = cameras_txt.exists() or cameras_bin.exists()
    has_images = images_txt.exists() or images_bin.exists()
    if not (has_camera and has_images):
        return None

    num_images = 0
    num_points = 0

    if images_bin.exists():
        num_images = _read_bin_count(images_bin)
    elif images_txt.exists():
        num_images = _count_text_entries(images_txt, kind='images')

    if points_bin.exists():
        num_points = _read_bin_count(points_bin)
    elif points_txt.exists():
        num_points = _count_text_entries(points_txt, kind='points')

    return {
        'path': model_dir,
        'name': model_dir.name,
        'num_images': num_images,
        'num_points': num_points,
        'has_text': cameras_txt.exists() and images_txt.exists(),
    }


def _select_best_sparse_submodel(colmap_dir):
    """Select the sparse submodel with the highest reconstruction coverage."""
    sparse_root = Path(colmap_dir) / 'sparse'
    if not sparse_root.exists():
        raise FileNotFoundError(f"No sparse reconstruction found in {colmap_dir}/sparse/.")

    candidate_dirs = [d for d in sorted(sparse_root.iterdir()) if d.is_dir()]
    candidates = []
    for model_dir in candidate_dirs:
        stats = _get_sparse_model_stats(model_dir)
        if stats is not None:
            candidates.append(stats)

    if not candidates:
        raise FileNotFoundError(
            f"No sparse reconstruction found in {colmap_dir}/sparse/. Run COLMAP first."
        )

    best = max(candidates, key=lambda c: (c['num_images'], c['num_points']))
    print(
        f"Selected sparse model '{best['name']}' "
        f"({best['num_images']} registered images, {best['num_points']} points3D)."
    )
    return best['path']


def ensure_text_format(colmap_dir):
    """Run colmap model_converter if sparse output is in binary format."""
    sparse_dir = _select_best_sparse_submodel(colmap_dir)

    if (sparse_dir / 'cameras.txt').exists() and (sparse_dir / 'images.txt').exists():
        return sparse_dir

    print(f"Converting binary COLMAP model to text format in {sparse_dir}...")
    try:
        subprocess.run(
            [
                'colmap', 'model_converter',
                '--input_path', str(sparse_dir),
                '--output_path', str(sparse_dir),
                '--output_type', 'TXT',
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        try:
            subprocess.run(
                [
                    'docker', 'run', '--rm',
                    '-v', f'{sparse_dir.resolve()}:/model',
                    '-w', '/model',
                    'colmap/colmap:latest',
                    'colmap', 'model_converter',
                    '--input_path', '/model',
                    '--output_path', '/model',
                    '--output_type', 'TXT',
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot convert binary model. Install COLMAP or use Docker. Error: {e}"
            )

    return sparse_dir


def main():
    parser = argparse.ArgumentParser(
        description='Convert COLMAP sparse reconstruction to ACE dataset format.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--colmap-dir', required=True,
        help='COLMAP series directory (contains sparse/ and images/).',
    )
    parser.add_argument(
        '--output-dir', required=True,
        help='Output directory for ACE dataset.',
    )
    parser.add_argument(
        '--train-ratio', type=float, default=0.8,
        help='Fraction of images for training (rest goes to test).',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducible train/test split.',
    )
    parser.add_argument(
        '--symlink', action='store_true', default=False,
        help='Create symlinks to images instead of copying them.',
    )

    args = parser.parse_args()

    colmap_dir = Path(args.colmap_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    images_dir = colmap_dir / 'images'

    if not images_dir.exists():
        print(f"Error: images directory not found at {images_dir}")
        sys.exit(1)

    sparse_dir = ensure_text_format(colmap_dir)

    cameras = parse_cameras_txt(sparse_dir / 'cameras.txt')
    images = parse_images_txt(sparse_dir / 'images.txt')

    print(f"Found {len(cameras)} camera(s) and {len(images)} registered image(s).")

    if not images:
        print("Error: no images found in COLMAP reconstruction.")
        sys.exit(1)

    random.seed(args.seed)
    indices = list(range(len(images)))
    random.shuffle(indices)

    n_train = max(1, int(len(images) * args.train_ratio))
    train_indices = set(indices[:n_train])

    for split in ('train', 'test'):
        for subdir in ('rgb', 'poses', 'calibration'):
            (output_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    stats = {'train': 0, 'test': 0, 'skipped': 0}

    for idx, img in enumerate(images):
        src_image = images_dir / img['name']

        if not src_image.exists():
            alt_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            stem = src_image.stem
            found = False
            for ext in alt_extensions:
                candidate = images_dir / (stem + ext)
                if candidate.exists():
                    src_image = candidate
                    found = True
                    break
            if not found:
                print(f"  Warning: image not found: {img['name']}, skipping.")
                stats['skipped'] += 1
                continue

        camera = cameras[img['camera_id']]
        c2w = colmap_to_cam2world(img['qw'], img['qx'], img['qy'], img['qz'],
                                  img['tx'], img['ty'], img['tz'])
        calib_value, calib_type = extract_calibration(camera)

        split = 'train' if idx in train_indices else 'test'
        base_name = Path(img['name']).stem

        rgb_dst = output_dir / split / 'rgb' / src_image.name
        if args.symlink:
            if rgb_dst.exists() or rgb_dst.is_symlink():
                rgb_dst.unlink()
            rgb_dst.symlink_to(src_image.resolve())
        else:
            shutil.copy2(str(src_image), str(rgb_dst))

        write_pose(output_dir / split / 'poses' / f'{base_name}.txt', c2w)
        write_calibration(output_dir / split / 'calibration' / f'{base_name}.txt',
                          calib_value, calib_type)

        stats[split] += 1

    print(f"\nConversion complete:")
    print(f"  Train images : {stats['train']}")
    print(f"  Test images  : {stats['test']}")
    if stats['skipped']:
        print(f"  Skipped      : {stats['skipped']}")
    print(f"  Output       : {output_dir}")


if __name__ == '__main__':
    main()
