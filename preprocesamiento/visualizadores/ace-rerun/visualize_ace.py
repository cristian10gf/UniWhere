#!/usr/bin/env python3
"""
Visualizador interactivo de resultados ACE usando Rerun.

Uso simplificado (solo nombre de serie):
    uv run visualize_ace.py --serie _merged

Uso completo con overrides:
    uv run visualize_ace.py --serie _merged --query-images foto1.jpg foto2.jpg
    uv run visualize_ace.py --colmap-dir /path/to/serie-1 --scene /path/to/ace \\
        --test-poses poses.txt
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
from ace_rerun.stats import compute_stats
from ace_rerun.viewer import export_ply, log_model_stats, log_query_results, log_to_rerun

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# DATA_ROOT: preprocesamiento/data/
# __file__ = .../preprocesamiento/visualizadores/ace-rerun/visualize_ace.py
DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
ACE_DIR_DEFAULT = Path(__file__).resolve().parent.parent.parent / "models" / "ace"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualizador interactivo de resultados ACE con Rerun",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--serie", type=str, default=None,
        help="Nombre de la serie (e.g. '_merged'). Auto-resuelve todos los paths "
             "desde DATA_ROOT (preprocesamiento/data/).",
    )

    pc_group = parser.add_argument_group("Nube de puntos (COLMAP)")
    pc_group.add_argument(
        "--point-cloud", type=Path, default=None,
        help="Path directo a un archivo PLY (e.g. dense/0/fused.ply)",
    )
    pc_group.add_argument(
        "--colmap-dir", type=Path, default=None,
        help="Directorio COLMAP para auto-detectar nube de puntos",
    )

    scene_group = parser.add_argument_group("Dataset ACE")
    scene_group.add_argument(
        "--scene", type=Path, default=None,
        help="Directorio de la escena ACE (con train/ y test/). "
             "Obligatorio si no se usa --serie.",
    )
    scene_group.add_argument(
        "--test-poses", type=Path, default=None,
        help="Archivo de resultados de test ACE (poses_*.txt)",
    )

    net_group = parser.add_argument_group("Nube ACE desde red (opcional)")
    net_group.add_argument(
        "--model", type=Path, default=None,
        help="Head entrenado (.pt) para extraer nube adicional de la red ACE "
             "y para --query-images",
    )
    net_group.add_argument(
        "--encoder", type=Path, default=None,
        help="Encoder pre-entrenado (ace_encoder_pretrained.pt)",
    )
    net_group.add_argument(
        "--ace-dir", type=Path, default=ACE_DIR_DEFAULT,
        help="Directorio fuente de ACE",
    )

    parser.add_argument(
        "--query-images", type=Path, nargs="+", default=None,
        help="Una o más imágenes query para relocalizar y mostrar en la escena",
    )
    parser.add_argument(
        "--max-points", type=int, default=500_000,
        help="Máximo de puntos a visualizar",
    )
    parser.add_argument(
        "--filter-depth", type=float, default=10.0,
        help="Filtrar puntos a más de N metros (solo red ACE)",
    )
    parser.add_argument(
        "--image-height", type=int, default=480,
        help="Altura de imagen para red ACE",
    )
    parser.add_argument(
        "--export-ply", type=Path, default=None,
        help="Exportar nube resultante a PLY (para CloudCompare)",
    )

    return parser


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    """
    Deriva paths desde --serie cuando no se dan explícitamente.
    Modifica args in-place y retorna args.
    """
    if args.serie is not None:
        serie = args.serie
        # Derivar colmap-dir si no se dio explícitamente
        if args.colmap_dir is None and args.point_cloud is None:
            args.colmap_dir = DATA_ROOT / serie
        # Derivar scene si no se dio explícitamente
        if args.scene is None:
            args.scene = DATA_ROOT / serie / "ace"
        # Derivar test-poses si no se dio explícitamente
        if args.test_poses is None:
            candidate = DATA_ROOT / "output" / f"poses_{serie}.txt"
            if candidate.exists():
                args.test_poses = candidate
            else:
                _logger.info(f"Poses file no encontrado: {candidate} (omitido)")
        # Derivar model si no se dio explícitamente
        if args.model is None:
            candidate = DATA_ROOT / "output" / f"{serie}.pt"
            if candidate.exists():
                args.model = candidate
            else:
                _logger.info(f"Modelo no encontrado: {candidate} (omitido)")

    # Validar que hay scene
    if args.scene is None:
        _logger.error("Debes indicar --scene o --serie.")
        sys.exit(1)

    return args


def validate_args(args: argparse.Namespace) -> None:
    """Valida precondiciones después de resolve_paths. Llama a sys.exit si hay error."""
    if args.query_images and args.model is None:
        _logger.error(
            "--query-images requiere un modelo ACE (.pt). "
            "Indica --model o usa --serie con un modelo en data/output/<serie>.pt"
        )
        sys.exit(1)


def main():
    parser = build_parser()
    args = parser.parse_args()
    resolve_paths(args)
    validate_args(args)

    # --- Encoder por defecto si hay model ---
    if args.model and args.encoder is None:
        args.encoder = args.ace_dir / "ace_encoder_pretrained.pt"

    # --- Nube de puntos COLMAP ---
    pc_positions, pc_colors = None, None

    if args.point_cloud:
        if not args.point_cloud.exists():
            _logger.error(f"PLY no encontrado: {args.point_cloud}")
            sys.exit(1)
        pc_positions, pc_colors = load_ply(args.point_cloud)

    elif args.colmap_dir:
        if not args.colmap_dir.exists():
            _logger.error(f"Directorio COLMAP no encontrado: {args.colmap_dir}")
            sys.exit(1)
        pc_positions, pc_colors = find_colmap_point_cloud(args.colmap_dir)

    if pc_positions is not None:
        pc_positions, pc_colors = subsample_point_cloud(
            pc_positions, pc_colors, args.max_points
        )

    # --- Nube ACE desde red (opcional, solo si no se usan query images) ---
    ace_pc_positions, ace_pc_colors = None, None

    if args.model and not args.query_images:
        for name, path in [("model", args.model), ("encoder", args.encoder)]:
            if not path.exists():
                _logger.error(f"{name} no encontrado: {path}")
                sys.exit(1)

        train_dir = args.scene / "train"
        if not train_dir.exists():
            _logger.error(f"Train dir no encontrado: {train_dir}")
            sys.exit(1)

        ace_pc_positions, ace_pc_colors = extract_point_cloud_from_network(
            args.encoder, args.model, args.scene,
            args.ace_dir, args.filter_depth, args.max_points, args.image_height,
        )

    if pc_positions is None and ace_pc_positions is None:
        _logger.error(
            "Sin fuente de nube de puntos. Usa --point-cloud, --colmap-dir, o --model."
        )
        sys.exit(1)

    # --- Escena ACE ---
    if not args.scene.exists():
        _logger.error(f"Escena ACE no encontrada: {args.scene}")
        sys.exit(1)

    mapping_poses, _ = load_split_poses(args.scene, "train")
    test_poses, test_images = load_split_poses(args.scene, "test")

    ace_results = None
    if args.test_poses and args.test_poses.exists():
        _logger.info(f"Cargando resultados ACE: {args.test_poses}")
        ace_results = parse_ace_results(args.test_poses)

    calibration = load_calibration(args.scene)

    # --- Export PLY ---
    if args.export_ply and pc_positions is not None:
        export_ply(pc_positions, pc_colors, args.export_ply)

    # --- Rerun ---
    scene_name = args.scene.name
    rr.init(f"ace_visualizer/{scene_name}", spawn=True)

    log_to_rerun(
        pc_positions=pc_positions,
        pc_colors=pc_colors,
        mapping_poses=mapping_poses,
        test_poses=test_poses,
        test_images=test_images,
        ace_results=ace_results,
        calibration=calibration,
        ace_pc_positions=ace_pc_positions,
        ace_pc_colors=ace_pc_colors,
    )

    # --- Stats ---
    stats = compute_stats(
        scene_dir=args.scene,
        model_path=args.model,
        ace_results=ace_results,
    )
    log_model_stats(stats)

    # --- Query images ---
    if args.query_images:
        from ace_rerun.relocalization import QueryRelocalizer
        from ace_rerun.viewer import get_pinhole_params

        for path in args.query_images:
            if not path.exists():
                _logger.error(f"Imagen query no encontrada: {path}")
                sys.exit(1)

        for p in [args.model, args.encoder]:
            if not p.exists():
                _logger.error(f"Archivo no encontrado: {p}")
                sys.exit(1)

        relocalizer = QueryRelocalizer(
            encoder_path=args.encoder,
            head_path=args.model,
            ace_dir=args.ace_dir,
            image_height=args.image_height,
        )

        pinhole = get_pinhole_params(calibration)
        focal_length = pinhole[0] if pinhole else 500.0

        query_results = relocalizer.relocalize_images(
            image_paths=args.query_images,
            focal_length=focal_length,
        )
        log_query_results(query_results, calibration)

    _logger.info("Visualización lista. Usa el viewer de Rerun para explorar.")
    _logger.info("La barra temporal navega por los frames de test.")


if __name__ == "__main__":
    main()
