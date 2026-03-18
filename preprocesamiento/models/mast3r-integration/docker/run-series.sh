#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Uso: ./run-series.sh <serie> [opciones] [-- args_extra_mast3r]

Reconstruye una serie con MASt3R en Docker/CUDA y exporta un layout
compatible con el pipeline actual:
  - data/<serie>/database.db
  - data/<serie>/sparse/0/{cameras,images,points3D}.bin
  - data/<serie>/dense/0/fused.ply

Opciones:
  --data-root PATH             Carpeta base de datasets (default: preprocesamiento/data)
  --mode MODE                  automatic|advanced|shell (default: advanced)
  --cpu                        Fuerza ejecucion sin GPU
  --cpus N                     Numero de CPUs para Docker (default: todos)
  --matcher TYPE               sequential|exhaustive|vocab_tree (default por modo)
  --overlap N                  Overlap para matcher sequential (default: 20)
  --model-name NAME            Modelo HF (default: MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric)
  --weights PATH               Checkpoint local dentro del contenedor
  --conf-thr FLOAT             Umbral de confianza (default: 1.001)
  --pixel-tol N                Tolerancia pixel para matching denso (default: 0)
  --dense-matching             Activa matching denso en MASt3R
  --min-len-track N            Longitud minima de track (default: 5)
  --skip-geometric-verification Saltar verify_matches
  --use-glomap-mapper          Usar glomap mapper en vez de pycolmap mapper
  -h, --help                   Muestra esta ayuda

Ejemplos:
  ./run-series.sh edificio-a
  ./run-series.sh edificio-a --mode automatic --matcher sequential --overlap 25
  ./run-series.sh edificio-a --cpu --conf-thr 1.2
  ./run-series.sh edificio-a -- --min-len-track 7
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../../../..")
DATA_ROOT="${PROJECT_ROOT}/preprocesamiento/data"
SERIE=""
MODE="advanced"
FORCE_CPU=0
NUM_CPUS_OVERRIDE=""
MATCHER=""
OVERLAP=20
MODEL_NAME="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
WEIGHTS=""
CONF_THR="1.001"
PIXEL_TOL=0
DENSE_MATCHING=0
MIN_LEN_TRACK=5
SKIP_GEOM=0
USE_GLOMAP=0
EXTRA_ARGS=()

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --cpu)
            FORCE_CPU=1
            shift
            ;;
        --cpus)
            NUM_CPUS_OVERRIDE="$2"
            shift 2
            ;;
        --matcher)
            MATCHER="$2"
            shift 2
            ;;
        --overlap)
            OVERLAP="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --weights)
            WEIGHTS="$2"
            shift 2
            ;;
        --conf-thr)
            CONF_THR="$2"
            shift 2
            ;;
        --pixel-tol)
            PIXEL_TOL="$2"
            shift 2
            ;;
        --dense-matching)
            DENSE_MATCHING=1
            shift
            ;;
        --min-len-track)
            MIN_LEN_TRACK="$2"
            shift 2
            ;;
        --skip-geometric-verification)
            SKIP_GEOM=1
            shift
            ;;
        --use-glomap-mapper)
            USE_GLOMAP=1
            shift
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        -* )
            echo "Error: opcion desconocida '$1'"
            echo ""
            usage
            exit 1
            ;;
        *)
            if [ -z "$SERIE" ]; then
                SERIE="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [ -z "$SERIE" ]; then
    echo "Error: debes indicar el nombre de la serie."
    echo ""
    usage
    exit 1
fi

case "$MODE" in
    automatic|advanced|shell) ;;
    *)
        echo "Error: modo invalido '$MODE'. Usa automatic, advanced o shell."
        exit 1
        ;;
esac

DATA_ROOT=$(realpath "$DATA_ROOT")
SERIE_DIR="${DATA_ROOT}/${SERIE}"
IMAGES_DIR="${SERIE_DIR}/images"

if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: no se encontro '${IMAGES_DIR}'."
    echo "La estructura esperada es: preprocesamiento/data/<serie>/images/"
    exit 1
fi

if [ -z "$MATCHER" ]; then
    if [ "$MODE" = "automatic" ]; then
        MATCHER="sequential"
    else
        MATCHER="exhaustive"
    fi
fi

case "$MATCHER" in
    sequential|exhaustive|vocab_tree) ;;
    *)
        echo "Error: matcher invalido '$MATCHER'."
        exit 1
        ;;
esac

IMAGE_TAG="mast3r-pipeline:latest"
DOCKERFILE="${PROJECT_ROOT}/preprocesamiento/models/mast3r-integration/docker/Dockerfile"

if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
    echo "Construyendo imagen ${IMAGE_TAG}..."
    docker build -f "$DOCKERFILE" -t "$IMAGE_TAG" "$PROJECT_ROOT"
fi

if [ -n "$NUM_CPUS_OVERRIDE" ]; then
    NUM_CPUS="$NUM_CPUS_OVERRIDE"
else
    NUM_CPUS=$(nproc)
fi

GPU_ARGS=()
DEVICE="cuda"

if [ "$FORCE_CPU" -eq 1 ]; then
    DEVICE="cpu"
else
    if docker run --rm --gpus all --entrypoint nvidia-smi "$IMAGE_TAG" >/dev/null 2>&1; then
        GPU_ARGS+=(--gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
    elif docker run --rm --runtime=nvidia --entrypoint nvidia-smi "$IMAGE_TAG" >/dev/null 2>&1; then
        GPU_ARGS+=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
    else
        echo "WARN: Docker sin acceso a GPU; ejecutando en CPU."
        DEVICE="cpu"
    fi
fi

DOCKER_ARGS=(
    --rm
    --cpus="${NUM_CPUS}"
    --ipc=host
    --shm-size=16g
    -v "${DATA_ROOT}:/data"
    -w /data
)

if [ ${#GPU_ARGS[@]} -gt 0 ] && [ "$DEVICE" = "cuda" ]; then
    DOCKER_ARGS+=("${GPU_ARGS[@]}")
fi

if [ "$MODE" = "shell" ]; then
    exec docker run -it --entrypoint /bin/bash "${DOCKER_ARGS[@]}" "$IMAGE_TAG"
fi

MAPPING_ARGS=(
    --series-dir "/data/${SERIE}"
    --images-dir "/data/${SERIE}/images"
    --matcher "$MATCHER"
    --overlap "$OVERLAP"
    --model-name "$MODEL_NAME"
    --device "$DEVICE"
    --conf-thr "$CONF_THR"
    --pixel-tol "$PIXEL_TOL"
    --min-len-track "$MIN_LEN_TRACK"
)

if [ -n "$WEIGHTS" ]; then
    MAPPING_ARGS+=(--weights "$WEIGHTS")
fi

if [ "$DENSE_MATCHING" -eq 1 ]; then
    MAPPING_ARGS+=(--dense-matching)
fi

if [ "$SKIP_GEOM" -eq 1 ]; then
    MAPPING_ARGS+=(--skip-geometric-verification)
fi

if [ "$USE_GLOMAP" -eq 1 ]; then
    MAPPING_ARGS+=(--use-glomap-mapper)
fi

if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    MAPPING_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "========================================"
echo " MASt3R series runner"
echo "========================================"
echo "Serie        : ${SERIE}"
echo "Data root    : ${DATA_ROOT}"
echo "Modo         : ${MODE}"
echo "Matcher      : ${MATCHER}"
echo "Device       : ${DEVICE}"
echo "CPUs         : ${NUM_CPUS}"
echo "========================================"
echo ""

exec docker run "${DOCKER_ARGS[@]}" "$IMAGE_TAG" "${MAPPING_ARGS[@]}"
