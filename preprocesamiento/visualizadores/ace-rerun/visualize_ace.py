#!/usr/bin/env python3
"""
Visualizador interactivo de resultados ACE usando Rerun.

Carga la nube de puntos original de COLMAP (dense/sparse) con la que se
entrenó ACE, y la muestra junto con las poses de cámara y las predicciones
del modelo ACE para poder evaluar visualmente la calidad de la relocalización.

Uso:
    # Con nube PLY de COLMAP + resultados ACE
    uv run visualize_ace.py \\
        --point-cloud /path/to/dense/0/fused.ply \\
        --scene /path/to/ace-dataset \\
        --test-poses /path/to/poses_scene.txt

    # Auto-detectar nube desde directorio COLMAP
    uv run visualize_ace.py \\
        --colmap-dir /path/to/serie-1 \\
        --scene /path/to/ace-dataset \\
        --test-poses /path/to/poses_scene.txt

    # Extraer nube de puntos de la red ACE (alternativo)
    uv run visualize_ace.py \\
        --scene /path/to/ace-dataset \\
        --model /path/to/head.pt \\
        --encoder /path/to/ace_encoder_pretrained.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import rerun as rr

from ace_rerun.ace_extraction import extract_point_cloud_from_network
from ace_rerun.point_cloud import (
    find_colmap_point_cloud,
    load_ply,
    subsample_point_cloud,
)
from ace_rerun.poses import load_calibration, load_split_poses, parse_ace_results
from ace_rerun.viewer import export_ply, log_to_rerun

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

ACE_DIR_DEFAULT = Path(__file__).resolve().parent.parent.parent / "models" / "ace"


def main():
    parser = argparse.ArgumentParser(
        description="Visualizador interactivo de resultados ACE con Rerun",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    pc_group = parser.add_argument_group("Nube de puntos (COLMAP)")
    pc_group.add_argument(
        "--point-cloud", type=Path, default=None,
        help="Path directo a un archivo PLY (e.g. dense/0/fused.ply)",
    )
    pc_group.add_argument(
        "--colmap-dir", type=Path, default=None,
        help="Directorio COLMAP para auto-detectar nube de puntos "
             "(busca dense/0/fused.ply, luego sparse/0/points3D.*)",
    )

    scene_group = parser.add_argument_group("Dataset ACE")
    scene_group.add_argument(
        "--scene", type=Path, required=True,
        help="Directorio de la escena ACE (con train/ y test/)",
    )
    scene_group.add_argument(
        "--test-poses", type=Path, default=None,
        help="Archivo de resultados de test ACE (poses_*.txt)",
    )

    net_group = parser.add_argument_group("Nube ACE desde red (opcional)")
    net_group.add_argument(
        "--model", type=Path, default=None,
        help="Head entrenado (.pt) para extraer nube adicional de la red ACE",
    )
    net_group.add_argument(
        "--encoder", type=Path, default=None,
        help="Encoder pre-entrenado (ace_encoder_pretrained.pt)",
    )
    net_group.add_argument(
        "--ace-dir", type=Path, default=ACE_DIR_DEFAULT,
        help="Directorio fuente de ACE",
    )

    parser.add_argument("--max-points", type=int, default=1_000_000,
                        help="Máximo de puntos a visualizar")
    parser.add_argument("--filter-depth", type=float, default=10.0,
                        help="Filtrar puntos a más de N metros (solo red ACE)")
    parser.add_argument("--image-height", type=int, default=480,
                        help="Altura de imagen para red ACE")
    parser.add_argument("--export-ply", type=Path, default=None,
                        help="Exportar nube resultante a PLY (para CloudCompare)")

    args = parser.parse_args()

    # --- Load COLMAP point cloud ---
    pc_positions, pc_colors = None, None

    if args.point_cloud:
        if not args.point_cloud.exists():
            _logger.error(f"PLY not found: {args.point_cloud}")
            sys.exit(1)
        pc_positions, pc_colors = load_ply(args.point_cloud)

    elif args.colmap_dir:
        if not args.colmap_dir.exists():
            _logger.error(f"COLMAP dir not found: {args.colmap_dir}")
            sys.exit(1)
        pc_positions, pc_colors = find_colmap_point_cloud(args.colmap_dir)

    if pc_positions is not None:
        pc_positions, pc_colors = subsample_point_cloud(pc_positions, pc_colors, args.max_points)

    # --- Optional: extract ACE network point cloud ---
    ace_pc_positions, ace_pc_colors = None, None

    if args.model:
        if args.encoder is None:
            args.encoder = args.ace_dir / "ace_encoder_pretrained.pt"

        for name, path in [("model", args.model), ("encoder", args.encoder)]:
            if not path.exists():
                _logger.error(f"{name} not found: {path}")
                sys.exit(1)

        train_dir = args.scene / "train"
        if not train_dir.exists():
            _logger.error(f"Train dir not found: {train_dir}")
            sys.exit(1)

        ace_pc_positions, ace_pc_colors = extract_point_cloud_from_network(
            args.encoder, args.model, args.scene,
            args.ace_dir, args.filter_depth, args.max_points, args.image_height,
        )

    # Must have at least one point cloud source
    if pc_positions is None and ace_pc_positions is None:
        _logger.error(
            "No point cloud source specified. Use --point-cloud, --colmap-dir, or --model."
        )
        sys.exit(1)

    # --- Load scene data ---
    if not args.scene.exists():
        _logger.error(f"Scene dir not found: {args.scene}")
        sys.exit(1)

    mapping_poses, mapping_images = load_split_poses(args.scene, "train")
    test_poses, test_images = load_split_poses(args.scene, "test")

    ace_results = None
    if args.test_poses and args.test_poses.exists():
        _logger.info(f"Loading ACE results: {args.test_poses}")
        ace_results = parse_ace_results(args.test_poses)

    calibration = load_calibration(args.scene)

    # --- Export PLY if requested ---
    if args.export_ply and pc_positions is not None:
        export_ply(pc_positions, pc_colors, args.export_ply)

    # --- Rerun ---
    scene_name = args.scene.name
    rr.init(f"ace_visualizer/{scene_name}", spawn=True)

    log_to_rerun(
        pc_positions=pc_positions,
        pc_colors=pc_colors,
        mapping_poses=mapping_poses,
        mapping_images=mapping_images,
        test_poses=test_poses,
        test_images=test_images,
        ace_results=ace_results,
        calibration=calibration,
        ace_pc_positions=ace_pc_positions,
        ace_pc_colors=ace_pc_colors,
    )

    _logger.info("Visualization ready. Use Rerun viewer to explore.")
    _logger.info("Timeline at the bottom navigates through test frames.")


if __name__ == "__main__":
    main()
