#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${SCRIPT_DIR}/data"

usage() {
    cat <<'EOF'
Uso: ./run-oneformer3d.sh --data-dir <path> [opciones]

Pipeline completo: convierte COLMAP → S3DIS .npy → OneFormer3D inferencia → labeled.ply

Paso 1: colmap2oneformer3d.py  (COLMAP → scene.npy)
Paso 2: docker/run-inference.sh (scene.npy → labeled.ply)

Opciones:
  --data-dir PATH         Directorio de la serie COLMAP (default: data/<serie>)
  --serie NAME            Nombre de serie (alternativa a --data-dir, usa data/<serie>)
  --checkpoint PATH       Checkpoint .pth (default: data/weights/oneformer3d/s3dis.pth)
  --config PATH           Config dentro del contenedor (default: configs/oneformer3d_1xb2_s3dis-area-5.py)
  --mode dense|sparse|auto Tipo de reconstrucción COLMAP a usar (default: auto)
  --voxel-size FLOAT      Tamaño de voxel para downsampling (default: 0.05)
  --estimate-normals      Forzar re-estimación de normales
  --cpu                   Forzar modo CPU (sin GPU)
  --skip-convert          Saltar paso 1 (scene.npy ya existe)
  -h, --help              Mostrar esta ayuda

Ejemplos:
  ./run-oneformer3d.sh --serie campus-norte
  ./run-oneformer3d.sh --data-dir data/_merged --mode dense
  ./run-oneformer3d.sh --serie campus-norte --cpu --voxel-size 0.03
  ./run-oneformer3d.sh --data-dir data/serie-1 --skip-convert
EOF
}

DATA_DIR=""
SERIE=""
CHECKPOINT=""
CONFIG="configs/oneformer3d_1xb2_s3dis-area-5.py"
MODE="auto"
VOXEL_SIZE="0.05"
ESTIMATE_NORMALS=0
CPU_FLAG=""
SKIP_CONVERT=0

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --serie) SERIE="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --voxel-size) VOXEL_SIZE="$2"; shift 2 ;;
        --estimate-normals) ESTIMATE_NORMALS=1; shift ;;
        --cpu) CPU_FLAG="--cpu"; shift ;;
        --skip-convert) SKIP_CONVERT=1; shift ;;
        *)
            echo "Error: argumento desconocido '$1'"
            echo ""
            usage
            exit 1
            ;;
    esac
done

# Resolve data directory
if [ -n "$SERIE" ] && [ -z "$DATA_DIR" ]; then
    DATA_DIR="${DATA_ROOT}/${SERIE}"
fi

if [ -z "$DATA_DIR" ]; then
    echo "Error: se requiere --data-dir o --serie."
    echo ""
    usage
    exit 1
fi

DATA_DIR="$(realpath "$DATA_DIR")"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: directorio '$DATA_DIR' no existe."
    exit 1
fi

# Default checkpoint location
if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT="${SCRIPT_DIR}/data/weights/oneformer3d/s3dis.pth"
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: checkpoint no encontrado: ${CHECKPOINT}"
    echo ""
    echo "Descarga el checkpoint de S3DIS y colócalo en:"
    echo "  ${SCRIPT_DIR}/data/weights/oneformer3d/s3dis.pth"
    echo ""
    echo "Puedes descargarlo desde el repositorio de OneFormer3D:"
    echo "  https://github.com/oneformer3d/oneformer3d#pretrained-models"
    exit 1
fi

# Output paths
ONEFORMER_DIR="${DATA_DIR}/oneformer3d"
INPUT_DIR="${ONEFORMER_DIR}/input"
OUTPUT_DIR="${ONEFORMER_DIR}/output"
SCENE_NPY="${INPUT_DIR}/scene.npy"
LABELED_PLY="${OUTPUT_DIR}/labeled.ply"

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

echo "========================================"
echo " OneFormer3D Pipeline"
echo "========================================"
echo "  Data dir      : ${DATA_DIR}"
echo "  Checkpoint    : ${CHECKPOINT}"
echo "  Config        : ${CONFIG}"
echo "  Mode          : ${MODE}"
echo "  Voxel size    : ${VOXEL_SIZE}"
echo "========================================"
echo ""

# --- Step 1: COLMAP → S3DIS .npy ---
if [ "$SKIP_CONVERT" -eq 0 ]; then
    echo "── Paso 1: COLMAP → scene.npy ──"

    if ! command -v uv >/dev/null 2>&1; then
        echo "Error: 'uv' no encontrado. Instálalo desde https://docs.astral.sh/uv/"
        exit 1
    fi

    CONVERT_ARGS=(
        --colmap-dir "$DATA_DIR"
        --output-dir "$INPUT_DIR"
        --mode "$MODE"
        --voxel-size "$VOXEL_SIZE"
    )

    if [ "$ESTIMATE_NORMALS" -eq 1 ]; then
        CONVERT_ARGS+=(--estimate-normals)
    fi

    uv run "${SCRIPT_DIR}/scripts/colmap2oneformer3d.py" "${CONVERT_ARGS[@]}"
    echo ""
else
    echo "── Paso 1: saltado (--skip-convert) ──"
    if [ ! -f "$SCENE_NPY" ]; then
        echo "Error: --skip-convert pero ${SCENE_NPY} no existe."
        exit 1
    fi
    echo "  Usando: ${SCENE_NPY}"
    echo ""
fi

# --- Step 2: OneFormer3D inference ---
echo "── Paso 2: OneFormer3D inferencia ──"

INFERENCE_ARGS=(
    --input "$SCENE_NPY"
    --output "$LABELED_PLY"
    --checkpoint "$CHECKPOINT"
    --config "$CONFIG"
    --voxel-size "$VOXEL_SIZE"
)

if [ -n "$CPU_FLAG" ]; then
    INFERENCE_ARGS+=($CPU_FLAG)
fi

"${SCRIPT_DIR}/models/oneformer3d/docker/run-inference.sh" "${INFERENCE_ARGS[@]}"

echo ""
echo "========================================"
echo " OneFormer3D completado"
echo "========================================"
echo "  Entrada  : ${SCENE_NPY}"
echo "  Resultado: ${LABELED_PLY}"
echo ""
echo "Siguiente paso (NavGraph):"
echo "  # 1. Generar grafo con etiquetas por defecto:"
echo "  uv run scripts/oneformer3d2navgraph.py --input ${LABELED_PLY} --output-dir ${ONEFORMER_DIR}/navgraph/"
echo ""
echo "  # 2. Editar zone_labels.json para asignar nombres (bloques/plantas/salas/ubicaciones)"
echo ""
echo "  # 3. Re-generar grafo con etiquetas configuradas:"
echo "  uv run scripts/oneformer3d2navgraph.py --input ${LABELED_PLY} --output-dir ${ONEFORMER_DIR}/navgraph/ --labels ${ONEFORMER_DIR}/navgraph/zone_labels.json"
