# ACE Rerun Visualizer

Visualizador interactivo de resultados de ACE (Accelerated Coordinate Encoding) usando [Rerun](https://rerun.io/).

Carga la **nube de puntos original de COLMAP** (la reconstrucción densa/sparse con la que se entrenó ACE) y la muestra junto con las poses de cámara y las predicciones de ACE para evaluar visualmente la calidad de la relocalización.

## Requisitos

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) como gestor de dependencias
- GPU con CUDA (solo si se usa `--model` para extraer nube de la red ACE)

## Instalación

```bash
cd preprocesamiento/visualizadores/ace-rerun
uv sync
```

## Uso

### Caso principal: nube COLMAP + resultados ACE

```bash
# Con PLY directo (e.g. reconstrucción densa de COLMAP)
uv run visualize_ace.py \
    --point-cloud /path/to/serie-1/dense/0/fused.ply \
    --scene /path/to/serie-1/ace \
    --test-poses /path/to/poses_serie-1.txt

# Auto-detectar nube desde directorio COLMAP
uv run visualize_ace.py \
    --colmap-dir /path/to/serie-1 \
    --scene /path/to/serie-1/ace \
    --test-poses /path/to/poses_serie-1.txt
```

### Solo nube de puntos + poses de mapping (sin resultados de test)

```bash
uv run visualize_ace.py \
    --colmap-dir /path/to/serie-1 \
    --scene /path/to/serie-1/ace
```

### Extraer nube adicional desde la red ACE (opcional)

```bash
uv run visualize_ace.py \
    --colmap-dir /path/to/serie-1 \
    --scene /path/to/serie-1/ace \
    --model /path/to/serie-1.pt \
    --encoder ../../models/ace/ace_encoder_pretrained.pt \
    --test-poses /path/to/poses_serie-1.txt
```

Esto muestra **ambas** nubes en Rerun (COLMAP como `colmap_point_cloud` y la extraída de ACE como `ace_point_cloud`), permitiendo comparar.

### Exportar nube a PLY (para CloudCompare)

```bash
uv run visualize_ace.py \
    --colmap-dir /path/to/serie-1 \
    --scene /path/to/serie-1/ace \
    --export-ply nube_colmap.ply
```

## Parámetros

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--point-cloud` | Path directo a PLY (e.g. `dense/0/fused.ply`) | - |
| `--colmap-dir` | Directorio COLMAP (auto-detecta PLY o points3D) | - |
| `--scene` | Directorio dataset ACE (con `train/` y `test/`) | **Requerido** |
| `--test-poses` | Archivo de resultados ACE (`poses_*.txt`) | - |
| `--model` | Head entrenado (`.pt`) para nube adicional de red ACE | - |
| `--encoder` | Encoder pre-entrenado | Auto-detectado |
| `--ace-dir` | Directorio fuente de ACE | `../../models/ace` |
| `--max-points` | Máximo de puntos a visualizar | 1,000,000 |
| `--filter-depth` | Filtrar puntos lejanos (solo red ACE, metros) | 10.0 |
| `--export-ply` | Exportar nube a PLY para CloudCompare | - |

## Qué se visualiza en Rerun

- **Nube de puntos COLMAP** (`world/colmap_point_cloud`): la reconstrucción densa/sparse original coloreada
- **Nube de puntos ACE** (`world/ace_point_cloud`, opcional): extraída de la red entrenada
- **Cámaras de mapping**: frustums de las cámaras de entrenamiento
- **Poses estimadas**: frustums coloreados por error (verde=bueno, rojo=malo)
- **Trayectoria GT**: posiciones ground truth de test en azul
- **Timeline**: navega por los frames de test con la barra temporal
- **Métricas**: series temporales de error de rotación/traslación/inliers
- **Imagen de query**: se muestra en el frustum y en panel separado

## Detección automática de nubes (--colmap-dir)

Busca en este orden de prioridad:
1. `dense/0/fused.ply` (reconstrucción densa)
2. `dense/fused.ply`
3. `sparse/0/points3D.txt` (reconstrucción sparse)
4. `sparse/0/points3D.bin`

## Estructura esperada

```
serie-1/                        # --colmap-dir
├── dense/0/fused.ply           # Nube de puntos COLMAP
├── sparse/0/
│   ├── cameras.txt
│   ├── images.txt
│   └── points3D.bin
├── images/
└── ace/                        # --scene
    ├── train/
    │   ├── rgb/
    │   ├── poses/
    │   └── calibration/
    └── test/
        ├── rgb/
        ├── poses/
        └── calibration/
```
