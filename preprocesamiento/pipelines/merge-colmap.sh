#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Uso: ./merge-colmap.sh <serie1> <serie2> [serie3 ...] [opciones]

Crea una reconstruccion COLMAP unificada reutilizando features y matches
de los COLMAPs individuales (enfoque hibrido con database_merger).

Pasos:
  1. Merge secuencial de bases de datos (database_merger)
  2. Symlinks de imagenes de todas las series
  3. Matching cross-serie (solo pares nuevos entre series distintas)
  4. Mapper unificado (reconstruccion SfM sobre la BD completa)
  5. (Opcional) Reconstruccion densa

Opciones:
  --data-root PATH      Carpeta base de datos (default: preprocesamiento/data)
  --output-name NAME    Nombre de la carpeta de salida (default: _merged)
  --matcher TYPE        vocab_tree | exhaustive (default: vocab_tree)
  --vocab-tree PATH     Ruta al archivo de vocabulary tree (relativa a data-root)
  --with-dense          Incluir reconstruccion densa
  --cpu                 Forzar modo CPU
  --skip-to STAGE       Reanudar desde: merge|symlinks|matching|mapper|dense
  -h, --help            Mostrar esta ayuda

Ejemplos:
  ./merge-colmap.sh serie-1 serie-2 serie-3
  ./merge-colmap.sh serie-1 serie-2 --matcher exhaustive
  ./merge-colmap.sh serie-1 serie-2 --matcher vocab_tree --vocab-tree _merged/vocab_tree.bin
  ./merge-colmap.sh serie-1 serie-2 --with-dense
  ./merge-colmap.sh serie-1 serie-2 --skip-to matching

Requisitos:
  - Los COLMAPs individuales deben haberse ejecutado previamente
  - Las imagenes deben tener nombres unicos entre series (prefijo de serie)
  - Para vocab_tree_matcher: descargar vocab_tree y especificarlo con --vocab-tree
    (https://demuc.de/colmap/)
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
DATA_ROOT="${PROJECT_ROOT}/preprocesamiento/data"
OUTPUT_NAME="_merged"
MATCHER="vocab_tree"
VOCAB_TREE=""
WITH_DENSE=0
FORCE_CPU=0
SKIP_TO=""
SERIES=()

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
        --output-name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --matcher)
            MATCHER="$2"
            shift 2
            ;;
        --vocab-tree)
            VOCAB_TREE="$2"
            shift 2
            ;;
        --with-dense)
            WITH_DENSE=1
            shift
            ;;
        --cpu)
            FORCE_CPU=1
            shift
            ;;
        --skip-to)
            SKIP_TO="$2"
            shift 2
            ;;
        -*)
            echo "Error: opcion desconocida '$1'"
            echo ""
            usage
            exit 1
            ;;
        *)
            SERIES+=("$1")
            shift
            ;;
    esac
done

if [ "${#SERIES[@]}" -lt 2 ]; then
    echo "Error: se necesitan al menos 2 series para hacer merge."
    echo ""
    usage
    exit 1
fi

case "$MATCHER" in
    vocab_tree|exhaustive) ;;
    *)
        echo "Error: matcher invalido '$MATCHER'. Usa 'vocab_tree' o 'exhaustive'."
        exit 1
        ;;
esac

DATA_ROOT=$(realpath "$DATA_ROOT")
MERGED_DIR="${DATA_ROOT}/${OUTPUT_NAME}"

if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: la carpeta de datos '$DATA_ROOT' no existe."
    exit 1
fi

for serie in "${SERIES[@]}"; do
    if [ ! -f "${DATA_ROOT}/${serie}/database.db" ]; then
        echo "Error: '${DATA_ROOT}/${serie}/database.db' no existe."
        echo "Ejecuta COLMAP sobre '$serie' primero."
        exit 1
    fi
    if [ ! -d "${DATA_ROOT}/${serie}/images" ]; then
        echo "Error: '${DATA_ROOT}/${serie}/images' no existe."
        exit 1
    fi
done

# --- Docker image ---
select_colmap_image() {
    if docker image inspect colmap:latest >/dev/null 2>&1; then
        echo "colmap:latest"
    else
        echo "No se encontro la imagen local colmap:latest. Descargando la oficial..." >&2
        docker pull colmap/colmap:latest >/dev/null
        echo "colmap/colmap:latest"
    fi
}

configure_gpu_args() {
    local image="$1"

    GPU_ARGS=()
    USE_GPU=0

    if [ "$FORCE_CPU" -eq 1 ]; then
        return
    fi

    if docker run --rm --gpus all "$image" nvidia-smi >/dev/null 2>&1; then
        GPU_ARGS+=(--gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
        USE_GPU=1
        return
    fi

    if docker run --rm --runtime=nvidia "$image" nvidia-smi >/dev/null 2>&1; then
        GPU_ARGS+=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
        USE_GPU=1
    fi
}

COLMAP_IMAGE=$(select_colmap_image)
configure_gpu_args "$COLMAP_IMAGE"

NUM_CPUS=$(nproc)
DOCKER_BASE=(
    --rm
    --cpus="${NUM_CPUS}"
    --ipc=host
    --shm-size=16g
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${DATA_ROOT}:/data"
    -w /data
)

if [ ${#GPU_ARGS[@]} -gt 0 ]; then
    DOCKER_BASE+=("${GPU_ARGS[@]}")
fi

run_colmap() {
    docker run "${DOCKER_BASE[@]}" "${COLMAP_IMAGE}" colmap "$@"
}

# Stage control for --skip-to
REACHED_STAGE=0
should_run() {
    local stage="$1"
    if [ -z "$SKIP_TO" ]; then
        return 0
    fi
    if [ "$stage" = "$SKIP_TO" ]; then
        REACHED_STAGE=1
    fi
    if [ "$REACHED_STAGE" -eq 1 ]; then
        return 0
    fi
    echo "  Saltando etapa: $stage"
    return 1
}

echo "========================================"
echo " Merge COLMAP (hibrido)"
echo "========================================"
echo "Series       : ${SERIES[*]}"
echo "Matcher      : $MATCHER"
echo "Salida       : $MERGED_DIR"
echo "Dense        : $([ "$WITH_DENSE" -eq 1 ] && echo "si" || echo "no")"
echo "GPU          : $USE_GPU"
[ -n "$SKIP_TO" ] && echo "Reanudando   : $SKIP_TO"
echo "========================================"
echo ""

mkdir -p "$MERGED_DIR"

# =============================================
# PASO 1: Merge secuencial de bases de datos
# =============================================
if should_run "merge"; then
    echo "[1/4] Merge de bases de datos..."

    TMP_DIR=$(mktemp -d)
    trap 'rm -rf "$TMP_DIR"' EXIT

    cp "${DATA_ROOT}/${SERIES[0]}/database.db" "${TMP_DIR}/current.db"
    echo "  Base: ${SERIES[0]}"

    for ((i = 1; i < ${#SERIES[@]}; i++)); do
        serie="${SERIES[$i]}"
        echo "  + Merging: $serie"

        cp "${TMP_DIR}/current.db" "${TMP_DIR}/merged_input.db"
        cp "${DATA_ROOT}/${serie}/database.db" "${TMP_DIR}/merge_target.db"

        docker run --rm \
            -v "${TMP_DIR}:/tmp_merge" \
            -w /tmp_merge \
            "${COLMAP_IMAGE}" \
            colmap database_merger \
                --database_path1 /tmp_merge/merged_input.db \
                --database_path2 /tmp_merge/merge_target.db \
                --merged_database_path /tmp_merge/current.db
    done

    cp "${TMP_DIR}/current.db" "${MERGED_DIR}/database.db"
    rm -rf "$TMP_DIR"
    trap - EXIT

    echo "  BD unificada: ${MERGED_DIR}/database.db"
    echo ""
fi

# =============================================
# PASO 2: Symlinks de imagenes
# =============================================
if should_run "symlinks"; then
    echo "[2/4] Creando symlinks de imagenes..."

    mkdir -p "${MERGED_DIR}/images"

    link_count=0
    for serie in "${SERIES[@]}"; do
        for img in "${DATA_ROOT}/${serie}/images/"*; do
            [ -f "$img" ] || continue
            base=$(basename "$img")
            target="${MERGED_DIR}/images/${base}"
            if [ -e "$target" ] || [ -L "$target" ]; then
                rm -f "$target"
            fi
            ln -s "$(realpath "$img")" "$target"
            ((link_count++))
        done
    done

    echo "  Symlinks creados: $link_count imagenes en ${MERGED_DIR}/images/"
    echo ""
fi

# =============================================
# PASO 3: Matching cross-serie
# =============================================
if should_run "matching"; then
    echo "[3/4] Matching cross-serie ($MATCHER)..."
    echo "  COLMAP saltara pares intra-serie que ya tienen matches."

    MATCH_ARGS=(
        --database_path "/data/${OUTPUT_NAME}/database.db"
        --SiftMatching.use_gpu "$USE_GPU"
        --SiftMatching.max_ratio 0.8
        --SiftMatching.max_num_matches 32768
    )

    if [ "$USE_GPU" -eq 1 ]; then
        MATCH_ARGS+=(--SiftMatching.gpu_index 0)
    fi

    if [ "$MATCHER" = "vocab_tree" ]; then
        if [ -n "$VOCAB_TREE" ]; then
            VOCAB_TREE_PATH="/data/${VOCAB_TREE}"
        elif [ -f "${MERGED_DIR}/vocab_tree.bin" ]; then
            VOCAB_TREE_PATH="/data/${OUTPUT_NAME}/vocab_tree.bin"
        else
            echo ""
            echo "Error: vocab_tree_matcher requiere un archivo de vocabulary tree."
            echo "Descarga uno de https://demuc.de/colmap/ y:"
            echo "  a) Colocalo en '${MERGED_DIR}/vocab_tree.bin', o"
            echo "  b) Usa --vocab-tree <ruta_relativa_a_data_root>"
            echo ""
            echo "Tamanos recomendados:"
            echo "  < 1000 imagenes : vocab_tree_flickr100K_words32K.bin"
            echo "  1K-10K imagenes : vocab_tree_flickr100K_words256K.bin"
            echo "  > 10K imagenes  : vocab_tree_flickr100K_words1M.bin"
            echo ""
            echo "Alternativa: usa --matcher exhaustive (mas lento pero sin archivo extra)."
            exit 1
        fi

        run_colmap vocab_tree_matcher \
            "${MATCH_ARGS[@]}" \
            --VocabTreeMatching.vocab_tree_path "$VOCAB_TREE_PATH" \
            --VocabTreeMatching.num_images 100
    else
        run_colmap exhaustive_matcher "${MATCH_ARGS[@]}"
    fi

    echo "  Matching cross-serie completado."
    echo ""
fi

# =============================================
# PASO 4: Mapper unificado
# =============================================
if should_run "mapper"; then
    echo "[4/4] Reconstruccion SfM unificada..."

    mkdir -p "${MERGED_DIR}/sparse"

    run_colmap mapper \
        --database_path "/data/${OUTPUT_NAME}/database.db" \
        --image_path "/data/${OUTPUT_NAME}/images" \
        --output_path "/data/${OUTPUT_NAME}/sparse"

    echo "  Reconstruccion sparse completada."
    echo ""
fi

# Best sparse model
BEST_SPARSE="${MERGED_DIR}/sparse/0"
if [ -d "${MERGED_DIR}/sparse" ]; then
    CANDIDATE=$(find "${MERGED_DIR}/sparse" -maxdepth 2 -name 'points3D.bin' -print0 2>/dev/null \
        | xargs -0 ls -s 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
    if [ -n "$CANDIDATE" ] && [ -f "$CANDIDATE" ]; then
        BEST_SPARSE="$(dirname "$CANDIDATE")"
    fi
fi
BEST_SPARSE_REL="${OUTPUT_NAME}/sparse/$(basename "$BEST_SPARSE")"

# =============================================
# PASO 5 (opcional): Reconstruccion densa
# =============================================
if [ "$WITH_DENSE" -eq 1 ] && should_run "dense"; then
    echo "[5] Reconstruccion densa..."

    mkdir -p "${MERGED_DIR}/dense/0"

    run_colmap image_undistorter \
        --image_path "/data/${OUTPUT_NAME}/images" \
        --input_path "/data/${BEST_SPARSE_REL}" \
        --output_path "/data/${OUTPUT_NAME}/dense/0" \
        --output_type COLMAP

    DENSE_ARGS=(
        --workspace_path "/data/${OUTPUT_NAME}/dense/0"
        --workspace_format COLMAP
        --PatchMatchStereo.max_image_size 3200
        --PatchMatchStereo.cache_size 16
        --PatchMatchStereo.geom_consistency 1
    )

    if [ "$USE_GPU" -eq 1 ]; then
        DENSE_ARGS+=(--PatchMatchStereo.gpu_index 0)
    fi

    run_colmap patch_match_stereo "${DENSE_ARGS[@]}"

    run_colmap stereo_fusion \
        --workspace_path "/data/${OUTPUT_NAME}/dense/0" \
        --workspace_format COLMAP \
        --input_type geometric \
        --output_path "/data/${OUTPUT_NAME}/dense/0/fused.ply"

    echo "  Reconstruccion densa completada."
    echo ""
fi

echo "========================================"
echo " Merge completado"
echo "========================================"
echo "BD unificada     : ${MERGED_DIR}/database.db"
echo "Imagenes         : ${MERGED_DIR}/images/ (symlinks)"
echo "Sparse model     : ${BEST_SPARSE}/"
[ "$WITH_DENSE" -eq 1 ] && echo "Dense cloud      : ${MERGED_DIR}/dense/0/fused.ply"
echo "========================================"
