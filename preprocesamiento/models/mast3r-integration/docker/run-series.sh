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
