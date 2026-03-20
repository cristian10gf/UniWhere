# MASt3R-SfM Real Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reemplazar la integración "toy" COLMAP-mapper de MASt3R por el pipeline real `sparse_global_alignment`, manteniendo salida compatible con el resto del pipeline (COLMAP texto + fused.ply).

**Architecture:** El nuevo script `mast3r_sfm.py` usa `load_images` + `make_pairs` + `sparse_global_alignment` de MASt3R-SfM directamente, sin pasar por `pycolmap.incremental_mapping`. El objeto `SparseGA` resultante se exporta a formato texto COLMAP (`cameras.txt`, `images.txt`, `points3D.txt`) y `dense/0/fused.ply`. Las intrínsecas estimadas en espacio de inferencia (512 px) se escalan a la resolución original de las imágenes. El Dockerfile y `run-series.sh` se actualizan para reflejar el nuevo script y eliminar los args del pipeline legacy.

**Tech Stack:** Python 3.10+, PyTorch, MASt3R (`sparse_global_alignment`, `make_pairs`, `load_images`), NumPy, PIL, Docker (nvcr.io/nvidia/pytorch:24.01-py3), uv (en Dockerfile para installs más rápidos)

---

## Mapa de archivos

| Acción | Archivo | Responsabilidad |
|--------|---------|-----------------|
| CREATE | `preprocesamiento/models/mast3r-integration/mast3r_sfm.py` | Script principal: carga imágenes, corre SfM, exporta COLMAP texto + PLY |
| CREATE | `preprocesamiento/models/mast3r-integration/tests/test_mast3r_sfm.py` | Tests de geometría y escritores de formato |
| MODIFY | `preprocesamiento/models/mast3r-integration/docker/Dockerfile` | COPY del nuevo script, uv para installs, nuevo ENTRYPOINT |
| MODIFY | `preprocesamiento/models/mast3r-integration/docker/run-series.sh` | Nuevos args (`--scene-graph`, `--niter1/2`, `--lr1/2`, `--min-conf-thr`, `--subsample`), eliminar args legacy |
| DELETE | `preprocesamiento/scripts/mast3r_reconstruct.py` | Script legacy eliminado |

---

## Task 1: Funciones de geometría y escritura COLMAP (núcleo testeable)

**Files:**
- Create: `preprocesamiento/models/mast3r-integration/mast3r_sfm.py`
- Create: `preprocesamiento/models/mast3r-integration/tests/test_mast3r_sfm.py`

### Contexto técnico

`sparse_global_alignment(filelist, pairs, cache_path, model, ...)`:
- **Primer argumento `imgs`** = lista de paths string (no los dicts cargados). Los dicts van en `pairs_in`.
- Devuelve `SparseGA` con:
  - `get_im_poses()` → `cam2w` tensor `[N, 4, 4]` (cámara→mundo, coordenadas de inferencia)
  - `get_focals()` → tensor `[N]` de focales en píxeles del espacio de inferencia (e.g., 512 px)
  - `get_principal_points()` → tensor `[N, 2]` (cx, cy) en espacio de inferencia
  - `get_sparse_pts3d()` → lista de tensors `[M_i, 3]` (puntos ancla por imagen)
  - `get_pts3d_colors()` → lista de arrays `[M_i, 3]` (colores 0–1 float)
  - `get_dense_pts3d(clean_depth=True, subsample=8)` → `(pts3d_list, depthmaps_list, confs_list)`

**Escala de intrínsecas**: Las focales y puntos principales estimados por `sparse_global_alignment` están en coordenadas del espacio de inferencia (imagen redimensionada). Para `cameras.txt` se escalan a la resolución original:
```
scale_x = orig_W / infer_W
fx_orig  = fx_infer * scale_x
cx_orig  = cx_infer * scale_x
cy_orig  = cy_infer * scale_y   # scale_y = orig_H / infer_H
```

**COLMAP world→camera**: `R_w2c = R_c2w.T`, `t_w2c = -R_c2w.T @ t_c2w`.
Cuaternión COLMAP: `(w, x, y, z)` — implementar en numpy puro (scipy no garantizado en contenedor).

**Importante — `points3D.txt`**: Los puntos ancla esparsos no tienen keypoints 2D registrados en `images.txt` (la segunda línea de cada imagen está vacía). Por ello, `points3D.txt` incluye los puntos 3D con track mínimo (image_id, punto_idx) pero el modelo resultante **no es cargable con `pycolmap.Reconstruction()`**. Esto es aceptable porque `colmap2ace.py` solo lee `cameras.txt`/`images.txt` y `colmap2oneformer3d.py` solo lee `fused.ply`.

- [ ] **Step 1.1: Escribir tests para `rot_to_quat_wxyz` y `c2w_to_colmap_pose`**

```python
# preprocesamiento/models/mast3r-integration/tests/test_mast3r_sfm.py
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from mast3r_sfm import rot_to_quat_wxyz, c2w_to_colmap_pose


def quat_to_rot(w, x, y, z) -> np.ndarray:
    """Reconstruye R desde cuaternión (numpy-only, sin scipy)."""
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)    ],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z),  2*(y*z - x*w)    ],
        [2*(x*z - y*w),     2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ])


def test_rot_to_quat_identity():
    w, x, y, z = rot_to_quat_wxyz(np.eye(3))
    assert abs(w - 1.0) < 1e-6
    assert abs(x) < 1e-6 and abs(y) < 1e-6 and abs(z) < 1e-6


def test_rot_to_quat_180_around_z():
    R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
    w, x, y, z = rot_to_quat_wxyz(R)
    R_back = quat_to_rot(w, x, y, z)
    assert np.allclose(R, R_back, atol=1e-5)


def test_rot_to_quat_roundtrip_arbitrary():
    # Rotar 45° alrededor de eje (1,1,1)/sqrt(3)
    theta = np.pi / 4
    ax = np.array([1, 1, 1]) / np.sqrt(3)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    w, x, y, z = rot_to_quat_wxyz(R)
    R_back = quat_to_rot(w, x, y, z)
    assert np.allclose(R, R_back, atol=1e-5)


def test_c2w_to_colmap_pose_identity():
    qw, qx, qy, qz, tx, ty, tz = c2w_to_colmap_pose(np.eye(4))
    assert abs(qw - 1.0) < 1e-6
    assert abs(tx) < 1e-6 and abs(ty) < 1e-6 and abs(tz) < 1e-6


def test_c2w_to_colmap_pose_translation():
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    _, _, _, _, tx, ty, tz = c2w_to_colmap_pose(c2w)
    assert abs(tx - (-1.0)) < 1e-6
    assert abs(ty - (-2.0)) < 1e-6
    assert abs(tz - (-3.0)) < 1e-6
```

- [ ] **Step 1.2: Ejecutar tests — deben fallar**

```bash
cd /home/worker-node-4/Documents/GitHub/UniWhere
python -m pytest preprocesamiento/models/mast3r-integration/tests/test_mast3r_sfm.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'mast3r_sfm'`

- [ ] **Step 1.3: Crear `mast3r_sfm.py` con las funciones de geometría**

```python
# preprocesamiento/models/mast3r-integration/mast3r_sfm.py
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
```

- [ ] **Step 1.4: Ejecutar tests — deben pasar**

```bash
cd /home/worker-node-4/Documents/GitHub/UniWhere
python -m pytest preprocesamiento/models/mast3r-integration/tests/test_mast3r_sfm.py -v
```

Expected: 5 tests passed

- [ ] **Step 1.5: Commit**

```bash
git add preprocesamiento/models/mast3r-integration/mast3r_sfm.py \
        preprocesamiento/models/mast3r-integration/tests/test_mast3r_sfm.py
git commit -m "feat(mast3r-sfm): geometry helpers rot_to_quat_wxyz y c2w_to_colmap_pose con tests"
```

---

## Task 2: Escritores de formato COLMAP texto

**Files:**
- Modify: `preprocesamiento/models/mast3r-integration/mast3r_sfm.py`
- Modify: `preprocesamiento/models/mast3r-integration/tests/test_mast3r_sfm.py`

- [ ] **Step 2.1: Agregar tests para todos los escritores**

```python
# Agregar al final de tests/test_mast3r_sfm.py
import tempfile
from mast3r_sfm import write_cameras_txt, write_images_txt, write_points3d_txt, write_ply


def test_write_cameras_txt_shared():
    focals = [800.0, 800.0]
    pps = [[960.0, 540.0], [960.0, 540.0]]
    shapes = [(1080, 1920), (1080, 1920)]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cameras.txt"
        write_cameras_txt(p, focals, pps, shapes, shared_intrinsics=True)
        lines = [l for l in p.read_text().splitlines() if not l.startswith('#') and l.strip()]
        assert len(lines) == 1
        parts = lines[0].split()
        assert parts[0] == '1'
        assert parts[1] == 'PINHOLE'
        assert parts[2] == '1920'   # width
        assert parts[3] == '1080'   # height
        assert abs(float(parts[4]) - 800.0) < 0.01   # fx
        assert abs(float(parts[6]) - 960.0) < 0.01   # cx
        assert abs(float(parts[7]) - 540.0) < 0.01   # cy


def test_write_cameras_txt_per_image():
    focals = [800.0, 820.0]
    pps = [[960.0, 540.0], [970.0, 545.0]]
    shapes = [(1080, 1920), (1080, 1920)]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cameras.txt"
        write_cameras_txt(p, focals, pps, shapes, shared_intrinsics=False)
        lines = [l for l in p.read_text().splitlines() if not l.startswith('#') and l.strip()]
        assert len(lines) == 2
        assert lines[0].split()[0] == '1'
        assert lines[1].split()[0] == '2'


def test_write_images_txt_identity_pose():
    c2w = np.eye(4)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "images.txt"
        write_images_txt(p, [c2w], ["frame000000.jpg"], shared_intrinsics=True)
        lines = [l for l in p.read_text().splitlines()
                 if not l.startswith('#') and l.strip()]
        assert len(lines) == 2  # pose line + empty points line
        parts = lines[0].split()
        assert parts[0] == '1'
        assert abs(float(parts[1]) - 1.0) < 1e-6   # qw
        assert abs(float(parts[5]) - 0.0) < 1e-6   # tx
        assert parts[9] == 'frame000000.jpg'


def test_write_points3d_txt_filters_nan():
    pts = [np.array([[float('nan'), 0, 0], [1.0, 2.0, 3.0]])]
    cols = [np.array([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]])]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "points3D.txt"
        n = write_points3d_txt(p, pts, cols)
        assert n == 1   # solo el punto finito
        data = [l for l in p.read_text().splitlines() if not l.startswith('#') and l.strip()]
        assert len(data) == 1


def test_write_ply_confidence_filter():
    pts = [np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])]  # shape (1, 2, 3)
    confs = [np.array([[0.5, 2.0]])]   # primer punto bajo umbral, segundo pasa
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fused.ply"
        n = write_ply(p, pts, confs, colors_list=None, min_conf_thr=1.5)
        assert n == 1
        content = p.read_text()
        assert "element vertex 1" in content
```

- [ ] **Step 2.2: Ejecutar tests — deben fallar**

```bash
cd /home/worker-node-4/Documents/GitHub/UniWhere
python -m pytest preprocesamiento/models/mast3r-integration/tests/test_mast3r_sfm.py \
  -k "write" -v 2>&1 | tail -15
```

Expected: `ImportError` o `AttributeError` (funciones no existen aún)

- [ ] **Step 2.3: Implementar los 4 escritores en `mast3r_sfm.py`**

Agregar después de `c2w_to_colmap_pose`:

```python
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


def write_points3d_txt(
    path: Path,
    pts3d_list: list,
    colors_list: list,
) -> int:
    """Escribe points3D.txt desde puntos ancla esparsos por imagen.

    Cada punto recibe un ID único. El track es mínimo (image_id, point_idx).
    Retorna el número de puntos escritos.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    pid = 1
    for img_idx, (pts, cols) in enumerate(zip(pts3d_list, colors_list)):
        pts_np = pts.detach().cpu().numpy() if isinstance(pts, torch.Tensor) else np.asarray(pts)
        cols_np = np.asarray(cols)
        for j in range(len(pts_np)):
            x, y, z = pts_np[j]
            if not np.isfinite([x, y, z]).all():
                continue
            rgb = cols_np[j]
            if rgb.max() <= 1.0:
                r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            else:
                r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
            rows.append(
                f"{pid} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0 {img_idx + 1} {j}\n"
            )
            pid += 1

    with path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(rows)}\n")
        f.writelines(rows)

    return len(rows)


def write_ply(
    path: Path,
    pts3d_list: list,
    confs_list: list,
    colors_list: list | None,
    min_conf_thr: float,
) -> int:
    """Escribe dense/0/fused.ply ASCII desde puntos 3D densos filtrados por confianza."""
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices: list[tuple] = []

    for i, (pts, confs) in enumerate(zip(pts3d_list, confs_list)):
        pts_np = pts.detach().cpu().numpy() if isinstance(pts, torch.Tensor) else np.asarray(pts)
        conf_np = confs.detach().cpu().numpy() if isinstance(confs, torch.Tensor) else np.asarray(confs)
        pts_flat = pts_np.reshape(-1, 3)
        conf_flat = conf_np.reshape(-1)
        mask = (conf_flat >= min_conf_thr) & np.isfinite(pts_flat).all(axis=1)

        if colors_list is not None:
            cols = colors_list[i]
            cols_np = cols.detach().cpu().numpy() if isinstance(cols, torch.Tensor) else np.asarray(cols)
            cols_flat = cols_np.reshape(-1, 3)
            for xyz, rgb in zip(pts_flat[mask], cols_flat[mask]):
                factor = 255 if rgb.max() <= 1.0 else 1
                r, g, b = int(rgb[0] * factor), int(rgb[1] * factor), int(rgb[2] * factor)
                vertices.append((*xyz, r, g, b))
        else:
            for xyz in pts_flat[mask]:
                vertices.append((*xyz, 255, 255, 255))

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.9f} {v[1]:.9f} {v[2]:.9f} {v[3]} {v[4]} {v[5]}\n")

    return len(vertices)
```

- [ ] **Step 2.4: Ejecutar todos los tests**

```bash
cd /home/worker-node-4/Documents/GitHub/UniWhere
python -m pytest preprocesamiento/models/mast3r-integration/tests/test_mast3r_sfm.py -v
```

Expected: 10 tests passed

- [ ] **Step 2.5: Commit**

```bash
git add preprocesamiento/models/mast3r-integration/mast3r_sfm.py \
        preprocesamiento/models/mast3r-integration/tests/test_mast3r_sfm.py
git commit -m "feat(mast3r-sfm): escritores COLMAP texto (cameras, images, points3D, PLY) con tests"
```

---

## Task 3: Función principal `main()` y CLI

**Files:**
- Modify: `preprocesamiento/models/mast3r-integration/mast3r_sfm.py`

- [ ] **Step 3.1: Agregar `list_images`, `scale_intrinsics`, `adapt_outputs`, `parse_args` y `main`**

Agregar al final de `mast3r_sfm.py`:

```python
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

    # points3D.txt (puntos ancla esparsos)
    sparse_pts  = scene.get_sparse_pts3d()
    sparse_cols = scene.get_pts3d_colors()
    n_sparse = write_points3d_txt(sparse_dst / "points3D.txt", sparse_pts, sparse_cols)
    print(f"  points3D.txt: {n_sparse} puntos")

    # fused.ply (nube densa filtrada por confianza)
    print(f"Generando nube densa (subsample={subsample})...")
    pts3d_dense, _, confs = scene.get_dense_pts3d(clean_depth=True, subsample=subsample)
    n_dense = write_ply(
        dense_dst / "fused.ply",
        pts3d_dense,
        confs,
        colors_list=None,
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
```

- [ ] **Step 3.2: Verificar que todos los tests aún pasan**

```bash
cd /home/worker-node-4/Documents/GitHub/UniWhere
python -m pytest preprocesamiento/models/mast3r-integration/tests/test_mast3r_sfm.py -v
```

Expected: 10 tests passed

- [ ] **Step 3.3: Verificar que `--help` funciona sin MASt3R instalado**

```bash
python preprocesamiento/models/mast3r-integration/mast3r_sfm.py --help
```

Expected: imprime ayuda sin `ImportError`

- [ ] **Step 3.4: Commit**

```bash
git add preprocesamiento/models/mast3r-integration/mast3r_sfm.py
git commit -m "feat(mast3r-sfm): main() con scale_intrinsics, adapt_outputs y CLI completa"
```

---

## Task 4: Actualizar Dockerfile y run-series.sh; eliminar legacy

**Files:**
- Modify: `preprocesamiento/models/mast3r-integration/docker/Dockerfile`
- Modify: `preprocesamiento/models/mast3r-integration/docker/run-series.sh`
- Delete: `preprocesamiento/scripts/mast3r_reconstruct.py`

- [ ] **Step 4.1: Reescribir `Dockerfile`**

Contenido completo de `preprocesamiento/models/mast3r-integration/docker/Dockerfile`:

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv (gestor de paquetes más rápido que pip para builds repetidos)
# Fijar versión para builds reproducibles
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_UNMANAGED_INSTALL="/usr/local/bin" sh

# Copiar MASt3R y asegurar dust3r submodule
COPY preprocesamiento/models/mast3r /opt/mast3r
RUN if [ ! -f /opt/mast3r/dust3r/requirements.txt ]; then \
        rm -rf /opt/mast3r/dust3r && \
        git clone --recursive https://github.com/naver/dust3r /opt/mast3r/dust3r; \
    fi

# Instalar dependencias con uv --system
WORKDIR /opt/mast3r/dust3r
RUN uv pip install --system -r requirements.txt
RUN uv pip install --system -r requirements_optional.txt
RUN uv pip install --system "opencv-python==4.8.0.74"

# Compilar kernels CUDA RoPE (embeddings posicionales, acelera inferencia)
WORKDIR /opt/mast3r/dust3r/croco/models/curope/
RUN python setup.py build_ext --inplace

WORKDIR /opt/mast3r
RUN uv pip install --system -r requirements.txt

# Copiar script principal desde mast3r-integration/ (junto al Dockerfile)
COPY preprocesamiento/models/mast3r-integration/mast3r_sfm.py /opt/unwhere/mast3r_sfm.py

ENV PYTHONPATH=/opt/mast3r:/opt/mast3r/dust3r

ENTRYPOINT ["python", "/opt/unwhere/mast3r_sfm.py"]
```

- [ ] **Step 4.2: Reescribir `run-series.sh`**

Contenido completo de `preprocesamiento/models/mast3r-integration/docker/run-series.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Uso: ./run-series.sh <serie> [opciones]

Reconstruye una serie con MASt3R-SfM (sparse_global_alignment) en Docker/CUDA
y exporta layout compatible con el pipeline UniWhere:
  - data/<serie>/sparse/0/{cameras,images,points3D}.txt
  - data/<serie>/dense/0/fused.ply

Opciones:
    --data-root PATH       Carpeta base de datasets (default: preprocesamiento/data)
    --cpu                  Fuerza ejecución sin GPU
    --cpus N               CPUs para Docker (default: auto, reserva 2 cores)
    --threads N            Hilos BLAS/OpenMP dentro del contenedor (default: igual a --cpus)
    --shm-size SIZE        Shared memory Docker, ej: 32g (default: auto según RAM host)
    --scene-graph GRAPH    Estrategia de pares: logwin-N, swin-N, complete, oneref-N
                           Añadir -noncyclic para desactivar cierre de loop (default: logwin-7)
    --image-size N         Lado largo para inferencia MASt3R, ej: 512 o 384 (default: 512)
    --niter1 N             Iteraciones coarse alignment (default: 300)
    --niter2 N             Iteraciones fine alignment (default: 300)
    --lr1 FLOAT            Learning rate coarse (default: 0.07)
    --lr2 FLOAT            Learning rate fine (default: 0.01)
    --min-conf-thr FLOAT   Umbral confianza nube densa (default: 1.5)
    --subsample N          Submuestreo nube densa en px (default: 8)
    --model-name NAME      Modelo HuggingFace (default: MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric)
    --weights PATH         Checkpoint local en el contenedor (default: auto en /data/weights/mast3r)
    --shell                Abre shell interactivo en el contenedor (para depuración)
    -h, --help             Muestra esta ayuda

Ejemplos:
  ./run-series.sh salon9
  ./run-series.sh salon9 --scene-graph logwin-7
  ./run-series.sh salon9 --niter1 500 --niter2 500 --min-conf-thr 2.0
  ./run-series.sh salon9 --scene-graph swin-20-noncyclic
  ./run-series.sh salon9 --cpu --image-size 384
  ./run-series.sh salon9 --shell
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../../../..")
DATA_ROOT="${PROJECT_ROOT}/preprocesamiento/data"
SERIE=""
FORCE_CPU=0
NUM_CPUS_OVERRIDE=""
THREADS_OVERRIDE=""
SHM_SIZE_OVERRIDE=""
SCENE_GRAPH="logwin-7"
IMAGE_SIZE=512
NITER1=300
NITER2=300
LR1="0.07"
LR2="0.01"
MIN_CONF_THR="1.5"
SUBSAMPLE=8
MODEL_NAME="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
WEIGHTS=""
SHELL_MODE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)        usage; exit 0 ;;
        --data-root)      DATA_ROOT="$2";       shift 2 ;;
        --cpu)            FORCE_CPU=1;           shift   ;;
        --cpus)           NUM_CPUS_OVERRIDE="$2"; shift 2 ;;
        --threads)        THREADS_OVERRIDE="$2"; shift 2 ;;
        --shm-size)       SHM_SIZE_OVERRIDE="$2"; shift 2 ;;
        --scene-graph)    SCENE_GRAPH="$2";      shift 2 ;;
        --image-size)     IMAGE_SIZE="$2";        shift 2 ;;
        --niter1)         NITER1="$2";            shift 2 ;;
        --niter2)         NITER2="$2";            shift 2 ;;
        --lr1)            LR1="$2";               shift 2 ;;
        --lr2)            LR2="$2";               shift 2 ;;
        --min-conf-thr)   MIN_CONF_THR="$2";      shift 2 ;;
        --subsample)      SUBSAMPLE="$2";         shift 2 ;;
        --model-name)     MODEL_NAME="$2";        shift 2 ;;
        --weights)        WEIGHTS="$2";           shift 2 ;;
        --shell)          SHELL_MODE=1;           shift   ;;
        -*)
            echo "Error: opción desconocida '$1'"; echo; usage; exit 1 ;;
        *)
            [ -z "$SERIE" ] && SERIE="$1" || { echo "Error: argumento extra '$1'"; exit 1; }
            shift ;;
    esac
done

if [ -z "$SERIE" ] && [ "$SHELL_MODE" -eq 0 ]; then
    echo "Error: debes indicar el nombre de la serie."; echo; usage; exit 1
fi

DATA_ROOT=$(realpath "$DATA_ROOT")
SERIE_DIR="${DATA_ROOT}/${SERIE}"
IMAGES_DIR="${SERIE_DIR}/images"

if [ "$SHELL_MODE" -eq 0 ] && [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: no se encontró '${IMAGES_DIR}'."
    echo "Estructura esperada: preprocesamiento/data/<serie>/images/"
    exit 1
fi

IMAGE_TAG="mast3r-pipeline:latest"
DOCKERFILE="${PROJECT_ROOT}/preprocesamiento/models/mast3r-integration/docker/Dockerfile"

if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
    echo "Construyendo imagen ${IMAGE_TAG}..."
    docker build -f "$DOCKERFILE" -t "$IMAGE_TAG" "$PROJECT_ROOT"
fi

# ── CPUs ──────────────────────────────────────────────────────────────────────
if [ -n "$NUM_CPUS_OVERRIDE" ]; then
    NUM_CPUS="$NUM_CPUS_OVERRIDE"
else
    TOTAL_CPUS=$(nproc)
    NUM_CPUS=$TOTAL_CPUS
    [ "$TOTAL_CPUS" -gt 8 ]  && NUM_CPUS=$((TOTAL_CPUS - 2))
    [ "$NUM_CPUS"   -gt 20 ] && NUM_CPUS=20
fi
THREADS="${THREADS_OVERRIDE:-$NUM_CPUS}"

# ── Shared memory ─────────────────────────────────────────────────────────────
if [ -n "$SHM_SIZE_OVERRIDE" ]; then
    SHM_SIZE="$SHM_SIZE_OVERRIDE"
else
    MEM_TOTAL_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
    if   [ "$MEM_TOTAL_KB" -ge 50000000 ]; then SHM_SIZE="24g"
    elif [ "$MEM_TOTAL_KB" -ge 24000000 ]; then SHM_SIZE="16g"
    else                                        SHM_SIZE="8g"
    fi
fi

# ── GPU ───────────────────────────────────────────────────────────────────────
GPU_ARGS=()
DEVICE="cuda"
if [ "$FORCE_CPU" -eq 1 ]; then
    DEVICE="cpu"
elif docker run --rm --gpus all --entrypoint nvidia-smi "$IMAGE_TAG" >/dev/null 2>&1; then
    GPU_ARGS+=(--gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
elif docker run --rm --runtime=nvidia --entrypoint nvidia-smi "$IMAGE_TAG" >/dev/null 2>&1; then
    GPU_ARGS+=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
else
    echo "WARN: Docker sin GPU; ejecutando en CPU."
    DEVICE="cpu"
fi

# ── Weights ───────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS_HOST="${DATA_ROOT}/weights/mast3r/${MODEL_NAME}.pth"
[ -z "$WEIGHTS" ] && [ -f "$DEFAULT_WEIGHTS_HOST" ] && \
    WEIGHTS="/data/weights/mast3r/${MODEL_NAME}.pth"

# ── Resumen ───────────────────────────────────────────────────────────────────
echo "========================================"
echo " MASt3R-SfM series runner"
echo "========================================"
echo "Serie        : ${SERIE}"
echo "Data root    : ${DATA_ROOT}"
echo "Scene graph  : ${SCENE_GRAPH}"
echo "Image size   : ${IMAGE_SIZE}"
echo "Device       : ${DEVICE}"
echo "CPUs         : ${NUM_CPUS}  Threads: ${THREADS}"
echo "SHM size     : ${SHM_SIZE}"
echo "niter1 / lr1 : ${NITER1} / ${LR1}"
echo "niter2 / lr2 : ${NITER2} / ${LR2}"
echo "min-conf-thr : ${MIN_CONF_THR}"
echo "subsample    : ${SUBSAMPLE}"
[ -n "$WEIGHTS" ] && echo "Weights      : ${WEIGHTS}" || \
    echo "Weights      : naver/${MODEL_NAME} (HF auto-download)"
echo "========================================"
echo ""

DOCKER_ARGS=(
    --rm
    --cpus="${NUM_CPUS}"
    --ipc=host
    --shm-size="${SHM_SIZE}"
    -e OMP_NUM_THREADS="${THREADS}"
    -e OPENBLAS_NUM_THREADS="${THREADS}"
    -e MKL_NUM_THREADS="${THREADS}"
    -e NUMEXPR_NUM_THREADS="${THREADS}"
    -e VECLIB_MAXIMUM_THREADS="${THREADS}"
    -v "${DATA_ROOT}:/data"
    -w /data
)
[ ${#GPU_ARGS[@]} -gt 0 ] && DOCKER_ARGS+=("${GPU_ARGS[@]}")

if [ "$SHELL_MODE" -eq 1 ]; then
    exec docker run -it --entrypoint /bin/bash "${DOCKER_ARGS[@]}" "$IMAGE_TAG"
fi

SFM_ARGS=(
    --series-dir   "/data/${SERIE}"
    --images-dir   "/data/${SERIE}/images"
    --scene-graph  "$SCENE_GRAPH"
    --image-size   "$IMAGE_SIZE"
    --niter1       "$NITER1"
    --niter2       "$NITER2"
    --lr1          "$LR1"
    --lr2          "$LR2"
    --min-conf-thr "$MIN_CONF_THR"
    --subsample    "$SUBSAMPLE"
    --model-name   "$MODEL_NAME"
    --device       "$DEVICE"
)
[ -n "$WEIGHTS" ] && SFM_ARGS+=(--weights "$WEIGHTS")

exec docker run "${DOCKER_ARGS[@]}" "$IMAGE_TAG" "${SFM_ARGS[@]}"
```

- [ ] **Step 4.3: Eliminar script legacy**

```bash
git rm preprocesamiento/scripts/mast3r_reconstruct.py
```

- [ ] **Step 4.4: Verificar ayuda del nuevo `run-series.sh`**

```bash
chmod +x preprocesamiento/models/mast3r-integration/docker/run-series.sh
bash preprocesamiento/models/mast3r-integration/docker/run-series.sh --help
```

Expected: ayuda con `--scene-graph`, `--niter1`, `--niter2`, etc. Sin flags legacy (`--matcher`, `--overlap`, `--conf-thr`, `--dense-matching`, `--use-glomap-mapper`).

- [ ] **Step 4.5: Commit**

```bash
git add preprocesamiento/models/mast3r-integration/docker/Dockerfile \
        preprocesamiento/models/mast3r-integration/docker/run-series.sh
git commit -m "feat(mast3r-sfm): Dockerfile uv + run-series.sh con args MASt3R-SfM; eliminar mast3r_reconstruct.py legacy"
```

---

## Task 5: Build Docker e integración smoke test

**Files:** ninguno nuevo

- [ ] **Step 5.1: Forzar rebuild de la imagen Docker**

```bash
docker rmi mast3r-pipeline:latest 2>/dev/null || true
docker build \
  -f preprocesamiento/models/mast3r-integration/docker/Dockerfile \
  -t mast3r-pipeline:latest \
  . 2>&1 | tee /tmp/mast3r_sfm_build.log
echo "Build exit code: $?"
```

Expected: exit code 0. Si falla, leer `/tmp/mast3r_sfm_build.log` para diagnosticar.

- [ ] **Step 5.2: Verificar `--help` dentro del contenedor**

```bash
docker run --rm mast3r-pipeline:latest --help
```

Expected: imprime ayuda de `mast3r_sfm.py` sin errores.

- [ ] **Step 5.3: Crear subset de 50 imágenes para smoke test**

```bash
mkdir -p /tmp/smoke_test/salon9_smoke/images
ls /home/worker-node-4/Documents/GitHub/UniWhere/preprocesamiento/data/salon9/images/ \
  | head -50 \
  | xargs -I{} cp \
      /home/worker-node-4/Documents/GitHub/UniWhere/preprocesamiento/data/salon9/images/{} \
      /tmp/smoke_test/salon9_smoke/images/
echo "Imágenes en smoke test: $(ls /tmp/smoke_test/salon9_smoke/images/ | wc -l)"
```

- [ ] **Step 5.4: Ejecutar smoke test con iteraciones reducidas**

```bash
cd /home/worker-node-4/Documents/GitHub/UniWhere
bash preprocesamiento/models/mast3r-integration/docker/run-series.sh salon9_smoke \
    --data-root /tmp/smoke_test \
    --niter1 50 --niter2 50 \
    --scene-graph logwin-5 \
    --min-conf-thr 1.0 \
    2>&1 | tee /tmp/mast3r_sfm_smoke.log
echo "Smoke test exit code: $?"
```

- [ ] **Step 5.5: Verificar outputs del smoke test**

```bash
echo "=== cameras.txt ==="
cat /tmp/smoke_test/salon9_smoke/sparse/0/cameras.txt

echo "=== images.txt (primeras 6 líneas de datos) ==="
grep -v '^#' /tmp/smoke_test/salon9_smoke/sparse/0/images.txt | head -6

echo "=== Imágenes registradas ==="
grep -v '^#' /tmp/smoke_test/salon9_smoke/sparse/0/images.txt | grep -v '^$' | wc -l

echo "=== points3D.txt (header) ==="
head -5 /tmp/smoke_test/salon9_smoke/sparse/0/points3D.txt

echo "=== fused.ply ==="
head -10 /tmp/smoke_test/salon9_smoke/dense/0/fused.ply
wc -l /tmp/smoke_test/salon9_smoke/dense/0/fused.ply
```

Expected:
- `cameras.txt`: 1 línea con `1 PINHOLE <W> <H> <fx> <fx> <cx> <cy>` — dimensiones originales (1920×1080 o las reales de salon9)
- `images.txt`: 50 líneas de poses (una por imagen) más 50 líneas vacías
- `points3D.txt`: encabezado con `Number of points: N` donde N > 0
- `fused.ply`: cabecera PLY válida + puntos con `element vertex N` donde N > 0

- [ ] **Step 5.6: Commit final**

```bash
git add preprocesamiento/models/mast3r-integration/
git commit -m "feat(mast3r-sfm): smoke test verificado — pipeline completo MASt3R-SfM con sparse_global_alignment"
```

---

## Notas de implementación

### Escala de intrínsecas (orig vs inferencia)
`sparse_global_alignment` optimiza en el espacio de `image_size=512` px. Las focales y pp resultantes están en ese espacio. Se escalan a la resolución original con `scale_x = orig_W / infer_W` para que `cameras.txt` sea correcto para `colmap2ace.py`.

### Caching entre ejecuciones
`<series-dir>/mast3r/cache/` persiste entre corridas. Si se re-ejecuta el script con las mismas imágenes, `forward_mast3r` salta los pares ya procesados (`.pth` en disco). Para forzar re-inferencia, borrar el directorio cache manualmente.

### `points3D.txt` — limitaciones conocidas
Los puntos ancla esparsos no tienen keypoints 2D registrados. El modelo **no es cargable con `pycolmap.Reconstruction()`**. Esto es intencional y funcional para el pipeline actual (colmap2ace.py y colmap2oneformer3d.py no dependen de esta funcionalidad).

### `--shell` para depuración
`run-series.sh --shell` abre un bash interactivo en el contenedor con `/data` montado, útil para depurar sin reconstruir la imagen.

### Salida texto vs binario
`colmap2ace.py` y `colmap2oneformer3d.py` soportan `.txt` y `.bin`. El nuevo pipeline produce `.txt` directamente — no se necesita conversión adicional.
