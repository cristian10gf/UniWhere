#!/usr/bin/env python3
"""MASt3R-SfM: sparse_global_alignment → salida COLMAP texto + fused.ply."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rot_to_quat_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    """3×3 rotation matrix → COLMAP quaternion (w, x, y, z), numpy-only."""
    R = np.asarray(R, dtype=float)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(w), float(x), float(y), float(z)


def c2w_to_colmap_pose(
    c2w: np.ndarray,
) -> tuple[float, float, float, float, float, float, float]:
    """Cam-to-world 4×4 → (qw, qx, qy, qz, tx, ty, tz) COLMAP world-to-cam."""
    c2w = np.asarray(c2w, dtype=float)
    R_c2w = c2w[:3, :3]
    t_c2w = c2w[:3, 3]
    R_w2c = R_c2w.T
    t_w2c = -R_c2w.T @ t_c2w
    qw, qx, qy, qz = rot_to_quat_wxyz(R_w2c)
    return qw, qx, qy, qz, float(t_w2c[0]), float(t_w2c[1]), float(t_w2c[2])


# ---------------------------------------------------------------------------
# COLMAP text format writers
# ---------------------------------------------------------------------------

def write_cameras_txt(
    path: Path,
    focals: list[float],
    principal_points: list[list[float]],
    image_shapes: list[tuple[int, int]],  # (H, W) en resolución ORIGINAL
    shared_intrinsics: bool,
) -> None:
    """Escribe cameras.txt en formato PINHOLE con resolución original.

    Las intrínsecas (focales, pp) deben venir ya escaladas a la resolución original.
    Si shared_intrinsics=True, una sola cámara (id=1) para todas las imágenes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        if shared_intrinsics:
            f.write("# Number of cameras: 1\n")
            H, W = image_shapes[0]
            fx = float(focals[0])
            cx, cy = float(principal_points[0][0]), float(principal_points[0][1])
            f.write(f"1 PINHOLE {W} {H} {fx:.6f} {fx:.6f} {cx:.6f} {cy:.6f}\n")
        else:
            f.write(f"# Number of cameras: {len(focals)}\n")
            for i, (fx, pp, shape) in enumerate(zip(focals, principal_points, image_shapes)):
                H, W = shape
                cx, cy = float(pp[0]), float(pp[1])
                f.write(
                    f"{i + 1} PINHOLE {W} {H} "
                    f"{float(fx):.6f} {float(fx):.6f} {cx:.6f} {cy:.6f}\n"
                )


def write_images_txt(
    path: Path,
    im_poses_c2w: list[np.ndarray],
    image_names: list[str],
    shared_intrinsics: bool,
) -> None:
    """Escribe images.txt convirtiendo cam2world → world2cam (COLMAP convention).

    Nota: la segunda línea de cada imagen (POINTS2D) queda vacía porque no se
    registran keypoints 2D individuales. Esto significa que el modelo sparse/0/
    NO es cargable con pycolmap.Reconstruction(), pero sí funciona con
    colmap2ace.py y colmap2oneformer3d.py (ambos leen text directamente).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_names)}\n")
        for i, (c2w, name) in enumerate(zip(im_poses_c2w, image_names)):
            qw, qx, qy, qz, tx, ty, tz = c2w_to_colmap_pose(c2w)
            cam_id = 1 if shared_intrinsics else (i + 1)
            f.write(
                f"{i + 1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                f"{tx:.9f} {ty:.9f} {tz:.9f} {cam_id} {name}\n"
            )
            f.write("\n")  # línea vacía de puntos 2D


def deduplicate_sparse_pts3d(
    pts3d_list: list,
    colors_list: list,
    voxel_fraction: float = 0.002,
) -> tuple[np.ndarray, np.ndarray]:
    """Deduplica anchor points de get_sparse_pts3d() por voxel grid adaptativo.

    get_sparse_pts3d() acumula observaciones de TODOS los pares que involucran
    cada imagen (N_pares × N_píxeles_por_par puntos), resultando en millones de
    duplicados del mismo punto físico 3D. Esta función los colapsa a un único
    representante por celda de voxel, preservando toda la geometría.

    El tamaño del voxel se adapta automáticamente a la escala de la escena:
        voxel_size = diagonal_bbox × voxel_fraction
    Con voxel_fraction=0.002 (0.2%) la resolución es análoga a tener ~500 celdas
    por dimensión — suficiente para capturar detalles finos independientemente de
    si la escena mide cm o km.

    Retorna:
        pts_dedup  (N_unique, 3)  float64 — puntos únicos finitos
        cols_dedup (N_unique, 3)  float64 — colores correspondientes (0-1)
    """
    # Concatenar y filtrar NaN/Inf
    parts_pts, parts_cols = [], []
    for pts, cols in zip(pts3d_list, colors_list):
        pts_np  = pts.detach().cpu().numpy() if isinstance(pts, torch.Tensor) else np.asarray(pts, dtype=float)
        cols_np = np.asarray(cols, dtype=float)
        if cols_np.max() > 1.0:
            cols_np = cols_np / 255.0
        valid = np.isfinite(pts_np).all(axis=1)
        parts_pts.append(pts_np[valid])
        parts_cols.append(cols_np[valid])

    if not parts_pts or sum(len(p) for p in parts_pts) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)

    all_pts  = np.concatenate(parts_pts,  axis=0)
    all_cols = np.concatenate(parts_cols, axis=0)

    # Eliminar outliers extremos (percentil 99.9 de la norma)
    norms = np.linalg.norm(all_pts, axis=1)
    p999  = np.percentile(norms, 99.9)
    inlier = norms <= p999
    all_pts  = all_pts[inlier]
    all_cols = all_cols[inlier]

    if len(all_pts) == 0:
        return all_pts, all_cols

    # Tamaño de voxel adaptativo a la escala de la escena
    mins     = all_pts.min(axis=0)
    maxs     = all_pts.max(axis=0)
    diagonal = float(np.linalg.norm(maxs - mins))
    voxel_size = max(diagonal * voxel_fraction, 1e-9)

    # Índices de voxel por punto
    vi    = ((all_pts - mins) / voxel_size).astype(np.int64)
    shape = vi.max(axis=0).astype(np.int64) + 1

    # Índice lineal (int64 evita overflow para scenes grandes)
    linear = vi[:, 0] * (shape[1] * shape[2]) + vi[:, 1] * shape[2] + vi[:, 2]

    # Primer punto de cada voxel
    _, first = np.unique(linear, return_index=True)

    return all_pts[first], all_cols[first]


def write_points3d_txt(
    path: Path,
    pts3d_list: list,
    colors_list: list,
) -> int:
    """Escribe points3D.txt desde puntos 3D (ya deduplicados por voxel grid).

    Cada punto recibe un ID único. El track es mínimo (image_id, point_idx).
    ACE y OneFormer3D no leen este archivo; solo cameras.txt/images.txt y
    fused.ply respectivamente. Retorna el número de puntos escritos.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    all_pts: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    img_ids: list[np.ndarray] = []

    for img_idx, (pts, cols) in enumerate(zip(pts3d_list, colors_list)):
        pts_np = (pts.detach().cpu().numpy() if isinstance(pts, torch.Tensor)
                  else np.asarray(pts, dtype=np.float64))
        cols_np = np.asarray(cols, dtype=np.float64)
        valid = np.isfinite(pts_np).all(axis=1)
        pts_v = pts_np[valid]
        cols_v = cols_np[valid]
        scale = 255.0 if cols_v.max() <= 1.0 else 1.0
        rgb = (cols_v * scale).clip(0, 255).astype(np.int32)
        all_pts.append(pts_v)
        all_rgb.append(rgb)
        img_ids.append(np.full(len(pts_v), img_idx + 1, dtype=np.int32))

    if not all_pts:
        with path.open("w", encoding="utf-8") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write("# Number of points: 0\n")
        return 0

    pts_out = np.vstack(all_pts)
    rgb_out = np.vstack(all_rgb)
    iid_out = np.concatenate(img_ids)
    n = len(pts_out)
    pids = np.arange(1, n + 1, dtype=np.int32)
    j_idx = np.arange(n, dtype=np.int32)

    with path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {n}\n")
        # Vectorized formatting via numpy: one call per chunk avoids Python loop
        data = np.column_stack([
            pids, pts_out, rgb_out,
            np.zeros(n, dtype=np.float64),  # ERROR
            iid_out, j_idx,
        ])
        np.savetxt(f, data,
                   fmt="%d %.6f %.6f %.6f %d %d %d %.1f %d %d")

    return n


def write_ply(
    path: Path,
    pts3d_list: list,
    confs_list: list,
    colors_list: list | None,
    min_conf_thr: float,
) -> int:
    """Escribe dense/0/fused.ply ASCII desde puntos 3D densos filtrados por confianza.

    colors_list debe ser una lista de arrays (H, W, 3) o (H*W, 3) float [0,1]
    correspondientes a los píxeles de la imagen (scene.imgs), o None para blanco.
    Usa numpy vectorizado — evita loops Python sobre millones de puntos.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []

    for i, (pts, confs) in enumerate(zip(pts3d_list, confs_list)):
        pts_np = (pts.detach().cpu().numpy() if isinstance(pts, torch.Tensor)
                  else np.asarray(pts, dtype=np.float32)).reshape(-1, 3)
        conf_np = (confs.detach().cpu().numpy() if isinstance(confs, torch.Tensor)
                   else np.asarray(confs)).reshape(-1)
        mask = (conf_np >= min_conf_thr) & np.isfinite(pts_np).all(axis=1)

        all_pts.append(pts_np[mask])

        if colors_list is not None:
            c = colors_list[i]
            c_np = (c.detach().cpu().numpy() if isinstance(c, torch.Tensor)
                    else np.asarray(c, dtype=np.float32)).reshape(-1, 3)[mask]
            # scene.imgs están en float [0,1]; convertir a uint8
            scale = 255.0 if float(c_np.max()) <= 1.0 else 1.0
            all_cols.append((c_np * scale).clip(0, 255).astype(np.uint8))
        else:
            all_cols.append(np.full((int(mask.sum()), 3), 255, dtype=np.uint8))

    pts_out = np.vstack(all_pts) if all_pts else np.empty((0, 3), dtype=np.float32)
    cols_out = np.vstack(all_cols) if all_cols else np.empty((0, 3), dtype=np.uint8)
    total = len(pts_out)

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {total}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if total > 0:
            data = np.hstack([pts_out.astype(np.float64), cols_out.astype(np.float64)])
            np.savetxt(f, data, fmt="%.9f %.9f %.9f %d %d %d")

    return total


# ---------------------------------------------------------------------------
# Image listing
# ---------------------------------------------------------------------------

def list_images(images_dir: Path) -> list[str]:
    """Retorna rutas absolutas ordenadas de imágenes válidas en images_dir."""
    return sorted(
        str(p)
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def get_original_shapes(filelist: list[str]) -> list[tuple[int, int]]:
    """Obtiene (H, W) original de cada imagen con PIL (sin cargar píxeles completos)."""
    from PIL import Image  # noqa: PLC0415
    shapes = []
    for fp in filelist:
        with Image.open(fp) as im:
            w, h = im.size
            shapes.append((h, w))
    return shapes


def scale_intrinsics(
    focals_infer: list[float],
    pps_infer: list[list[float]],
    infer_shapes: list[tuple[int, int]],   # (H, W) resolución de inferencia
    orig_shapes: list[tuple[int, int]],    # (H, W) resolución original
) -> tuple[list[float], list[list[float]]]:
    """Escala focales y puntos principales del espacio de inferencia al original."""
    focals_orig = []
    pps_orig = []
    for fx, pp, (ih, iw), (oh, ow) in zip(focals_infer, pps_infer, infer_shapes, orig_shapes):
        sx = ow / iw
        sy = oh / ih
        focals_orig.append(fx * sx)
        pps_orig.append([pp[0] * sx, pp[1] * sy])
    return focals_orig, pps_orig


# ---------------------------------------------------------------------------
# Adaptación de salida al layout de UniWhere
# ---------------------------------------------------------------------------

def adapt_outputs(
    series_dir: Path,
    scene,                        # SparseGA instance
    image_names: list[str],       # filename sin directorio
    orig_shapes: list[tuple[int, int]],   # (H, W) originales
    infer_shapes: list[tuple[int, int]],  # (H, W) de inferencia (del tensor true_shape)
    min_conf_thr: float,
    subsample: int,
) -> None:
    """Exporta SparseGA al layout COLMAP texto de UniWhere."""
    sparse_dst = series_dir / "sparse" / "0"
    dense_dst  = series_dir / "dense"  / "0"

    if sparse_dst.exists():
        shutil.rmtree(sparse_dst)
    sparse_dst.mkdir(parents=True)
    dense_dst.mkdir(parents=True, exist_ok=True)

    # Intrínsecas en espacio de inferencia
    focals_infer = scene.get_focals().detach().cpu().tolist()
    pps_infer    = scene.get_principal_points().detach().cpu().tolist()

    # Escalar a resolución original para cameras.txt
    focals_orig, pps_orig = scale_intrinsics(
        focals_infer, pps_infer, infer_shapes, orig_shapes
    )

    # cameras.txt
    write_cameras_txt(
        sparse_dst / "cameras.txt", focals_orig, pps_orig, orig_shapes,
        shared_intrinsics=True,
    )
    print(f"  cameras.txt : {sparse_dst / 'cameras.txt'}")

    # images.txt  (poses en coordenadas de inferencia — independiente de resolución)
    im_poses = scene.get_im_poses().detach().cpu().numpy()  # [N, 4, 4]
    write_images_txt(
        sparse_dst / "images.txt",
        [im_poses[i] for i in range(len(im_poses))],
        image_names,
        shared_intrinsics=True,
    )
    print(f"  images.txt  : {sparse_dst / 'images.txt'} ({len(image_names)} imágenes)")

    # points3D.txt — deduplicar por voxel grid antes de escribir
    # get_sparse_pts3d() acumula puntos de TODOS los pares por imagen (N duplicados
    # del mismo punto físico). deduplicate_sparse_pts3d() los colapsa a un único
    # representante por celda espacial, preservando la geometría completa.
    sparse_pts  = scene.get_sparse_pts3d()
    sparse_cols = scene.get_pts3d_colors()
    print("Deduplicando puntos ancla esparsos (voxel grid)...")
    pts_dedup, cols_dedup = deduplicate_sparse_pts3d(sparse_pts, sparse_cols)
    n_sparse = write_points3d_txt(sparse_dst / "points3D.txt", [pts_dedup], [cols_dedup])
    print(f"  points3D.txt: {n_sparse} puntos únicos")

    # fused.ply (nube densa filtrada por confianza)
    # Los colores vienen de scene.imgs: list de (H, W, 3) float [0,1] — misma
    # ordenación pixel por pixel que los pts3d densos que genera make_pts3d().
    print(f"Generando nube densa (subsample={subsample})...")
    pts3d_dense, _, confs = scene.get_dense_pts3d(clean_depth=True, subsample=subsample)
    n_dense = write_ply(
        dense_dst / "fused.ply",
        pts3d_dense,
        confs,
        colors_list=scene.imgs,  # float [0,1] (H,W,3) — el MASt3R viz usa esto mismo
        min_conf_thr=min_conf_thr,
    )
    print(f"  fused.ply   : {n_dense} puntos (conf >= {min_conf_thr})")
    print(f"\nReconstrucción exportada a: {series_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MASt3R-SfM: reconstrucción con sparse_global_alignment → COLMAP texto.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--series-dir",  required=True, help="Directorio raíz de la serie.")
    p.add_argument("--images-dir",  required=True, help="Directorio con imágenes.")
    p.add_argument("--work-dir",    default="",
                   help="Directorio de trabajo (default: <series-dir>/mast3r).")
    p.add_argument("--model-name",  default="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    p.add_argument("--weights",     default="",
                   help="Checkpoint local; si vacío, descarga de HuggingFace.")
    p.add_argument("--device",      default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--image-size",  type=int, default=512,
                   help="Lado largo de imagen para inferencia (512 = default MASt3R).")
    p.add_argument("--scene-graph", default="logwin-7",
                   help=(
                       "Estrategia de pares: 'logwin-N', 'swin-N', 'complete', 'oneref-N'. "
                       "Añadir '-noncyclic' para desactivar cierre de loop. "
                       "Para video interior: logwin-7 (default)."
                   ))
    p.add_argument("--niter1",       type=int,   default=300,
                   help="Iteraciones coarse alignment (3D matching loss).")
    p.add_argument("--niter2",       type=int,   default=300,
                   help="Iteraciones fine alignment (2D reprojection loss).")
    p.add_argument("--lr1",          type=float, default=0.07,
                   help="Learning rate coarse.")
    p.add_argument("--lr2",          type=float, default=0.01,
                   help="Learning rate fine.")
    p.add_argument("--min-conf-thr", type=float, default=1.5,
                   help="Umbral de confianza mínimo para la nube densa (fused.ply).")
    p.add_argument("--subsample",    type=int,   default=8,
                   help="Submuestreo para nube densa (8 = cada 8 px). "
                        "Mismo valor se usa en SfM y en exportación.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    series_dir = Path(args.series_dir).resolve()
    images_dir = Path(args.images_dir).resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"images-dir no encontrado: {images_dir}")

    work_dir = Path(args.work_dir).resolve() if args.work_dir else (series_dir / "mast3r")
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = str(work_dir / "cache")

    # Importaciones pesadas dentro de main() para no romper tests en host sin MASt3R
    from dust3r.utils.image import load_images                         # noqa: PLC0415
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment     # noqa: PLC0415
    from mast3r.image_pairs import make_pairs                          # noqa: PLC0415
    from mast3r.model import AsymmetricMASt3R                          # noqa: PLC0415

    # --- Listar imágenes ---
    filelist = list_images(images_dir)
    if len(filelist) < 2:
        raise RuntimeError(f"Se necesitan al menos 2 imágenes, encontradas: {len(filelist)}")
    print(f"Imágenes encontradas: {len(filelist)}")

    # --- Formas originales (antes de resize) ---
    print("Leyendo dimensiones originales de imágenes...")
    orig_shapes = get_original_shapes(filelist)

    # --- Cargar imágenes redimensionadas para inferencia ---
    print(f"Cargando imágenes (image-size={args.image_size})...")
    imgs = load_images(filelist, size=args.image_size, verbose=False)

    # Formas en espacio de inferencia (H, W) — de true_shape de load_images
    infer_shapes = [
        (int(d["true_shape"][0][0]), int(d["true_shape"][0][1])) for d in imgs
    ]
    image_names = [Path(fp).name for fp in filelist]

    # --- Generar pares ---
    print(f"Generando pares (scene-graph={args.scene_graph})...")
    # NOTA: make_pairs recibe los dicts cargados (imgs), NO filelist
    pairs = make_pairs(imgs, scene_graph=args.scene_graph, symmetrize=True)
    print(f"  {len(pairs)} pares generados")
    if not pairs:
        raise RuntimeError("No se generaron pares de imágenes.")

    # --- Cargar modelo ---
    weights_path = args.weights or f"naver/{args.model_name}"
    print(f"Cargando modelo: {weights_path}")
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)

    # --- MASt3R-SfM ---
    print(
        f"Ejecutando sparse_global_alignment "
        f"(niter1={args.niter1}, niter2={args.niter2}, device={args.device})..."
    )
    # NOTA: primer arg es filelist (paths string), NO los dicts imgs.
    # pairs_in contiene los dicts; filelist sirve para resolve_instance en convert_dust3r_pairs_naming.
    scene = sparse_global_alignment(
        filelist,
        pairs,
        cache_dir,
        model,
        subsample=args.subsample,
        device=args.device,
        shared_intrinsics=True,
        lr1=args.lr1,
        niter1=args.niter1,
        lr2=args.lr2,
        niter2=args.niter2,
    )
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    print(f"Reconstrucción completada: {scene.n_imgs} imágenes")

    # --- Exportar ---
    adapt_outputs(
        series_dir, scene, image_names,
        orig_shapes, infer_shapes,
        args.min_conf_thr, args.subsample,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
