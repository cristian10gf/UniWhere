#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Uso: ./pipeline.sh [opciones]

Orquestador end-to-end: videos -> frames -> COLMAP -> merge -> ACE.
Ejecuta todo o partes del pipeline segun las opciones indicadas.

Modos de ejecucion:
  Pipeline completo (video a ACE):
    ./pipeline.sh --series video1.mp4 video2.mp4 --merge --run-ace

  Solo frames + COLMAP individual (sin merge ni ACE):
    ./pipeline.sh --series video1.mp4

  Solo merge de series ya procesadas:
    ./pipeline.sh --merge-only serie-1 serie-2 serie-3

  Solo conversion COLMAP->ACE + entrenamiento:
    ./pipeline.sh --ace-only _merged/ace

Opciones generales:
  --data-root PATH         Carpeta base de datos (default: preprocesamiento/data)
  --cpu                    Forzar modo CPU en todos los pasos
  -h, --help               Mostrar esta ayuda

Opciones de extraccion de frames:
  --videos-dir PATH        Carpeta con los videos de entrada (default: preprocesamiento/videos)
  --series FILE [FILE...]  Videos a procesar (nombres relativos a --videos-dir)
  --sample-fps FPS         FPS de muestreo para extraccion de frames (default: 2)

  Los frames se extraen con las dimensiones originales del video (sin redimensionar).

Opciones de COLMAP:
  --max-parallel N         Maximo de COLMAPs en paralelo (default: 1)
  --colmap-mode MODE       Modo COLMAP: automatic|advanced (default: advanced)

Opciones de merge:
  --merge                  Ejecutar merge de bases de datos tras COLMAP individual
  --merge-only S [S...]    Solo merge (sin extraccion ni COLMAP individual)
  --matcher TYPE           vocab_tree | exhaustive (default: vocab_tree)
  --vocab-tree PATH        Ruta al vocabulary tree (relativa a data-root)

Opciones de ACE:
  --run-ace                Ejecutar conversion + entrenamiento ACE tras merge
  --ace-only PATH          Solo ACE sobre datos ya convertidos (ruta relativa a data-root)
  --train-ratio RATIO      Ratio train/test para colmap2ace (default: 0.8)
  --ace-output PATH        Ruta de salida del modelo ACE (relativa a data-root, default: output/<nombre>.pt)

Opciones de OneFormer3D:
  --run-oneformer3d        Ejecutar segmentacion semantica 3D tras COLMAP/merge
  --oneformer3d-checkpoint PATH  Checkpoint .pth (default: data/weights/oneformer3d/s3dis.pth)
  --oneformer3d-config PATH      Config del modelo (default: configs/oneformer3d_1xb2_s3dis-area-5.py)

Opciones de NavGraph:
  --run-navgraph           Generar grafo de navegacion tras OneFormer3D (implica --run-oneformer3d)
  --navgraph-voxel-size F  Resolucion de grid para zonas (default: 0.1)
  --navgraph-min-area F    Area minima de zona en m² (default: 2.0)

Ejemplos:
  # Pipeline completo (videos en preprocesamiento/videos/ por defecto)
  ./pipeline.sh --series campus-norte.mp4 campus-sur.mp4 --merge --run-ace

  # Videos en otra carpeta
  ./pipeline.sh --videos-dir /ruta/a/videos --series campus-norte.mp4 --merge --run-ace

  # COLMAP paralelo sobre series existentes (sin extraccion de frames)
  ./pipeline.sh --series-names serie-1 serie-2

  # Solo merge
  ./pipeline.sh --merge-only serie-1 serie-2 serie-3

  # Solo ACE
  ./pipeline.sh --ace-only _merged/ace

  # Pipeline con segmentacion 3D y grafo de navegacion
  ./pipeline.sh --series campus.mp4 --merge --run-oneformer3d --run-navgraph

  # Con checkpoint personalizado
  ./pipeline.sh --series campus.mp4 --merge --run-oneformer3d --oneformer3d-checkpoint weights/custom.pth
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
DATA_ROOT="${PROJECT_ROOT}/preprocesamiento/data"
FORCE_CPU=0

# Frame extraction
VIDEOS_DIR="${PROJECT_ROOT}/preprocesamiento/videos"
VIDEO_FILES=()
SAMPLE_FPS=2

# COLMAP
MAX_PARALLEL=1
COLMAP_MODE="advanced"
SERIES_NAMES=()

# Merge
DO_MERGE=0
MERGE_ONLY=0
MERGE_SERIES=()
MATCHER="vocab_tree"
VOCAB_TREE=""

# ACE
DO_ACE=0
ACE_ONLY=""
TRAIN_RATIO=0.8
ACE_OUTPUT=""

# OneFormer3D
DO_ONEFORMER3D=0
ONEFORMER3D_CHECKPOINT=""
ONEFORMER3D_CONFIG=""

# NavGraph
DO_NAVGRAPH=0
NAVGRAPH_VOXEL_SIZE="0.1"
NAVGRAPH_MIN_AREA="2.0"

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
        --cpu)
            FORCE_CPU=1
            shift
            ;;
        --videos-dir)
            VIDEOS_DIR="$2"
            shift 2
            ;;
        --series)
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                VIDEO_FILES+=("$1")
                shift
            done
            ;;
        --series-names)
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                SERIES_NAMES+=("$1")
                shift
            done
            ;;
        --sample-fps)
            SAMPLE_FPS="$2"
            shift 2
            ;;
        --max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --colmap-mode)
            COLMAP_MODE="$2"
            shift 2
            ;;
        --merge)
            DO_MERGE=1
            shift
            ;;
        --merge-only)
            MERGE_ONLY=1
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                MERGE_SERIES+=("$1")
                shift
            done
            ;;
        --matcher)
            MATCHER="$2"
            shift 2
            ;;
        --vocab-tree)
            VOCAB_TREE="$2"
            shift 2
            ;;
        --run-ace)
            DO_ACE=1
            shift
            ;;
        --ace-only)
            ACE_ONLY="$2"
            shift 2
            ;;
        --train-ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        --ace-output)
            ACE_OUTPUT="$2"
            shift 2
            ;;
        --run-oneformer3d)
            DO_ONEFORMER3D=1
            shift
            ;;
        --oneformer3d-checkpoint)
            ONEFORMER3D_CHECKPOINT="$2"
            shift 2
            ;;
        --oneformer3d-config)
            ONEFORMER3D_CONFIG="$2"
            shift 2
            ;;
        --run-navgraph)
            DO_NAVGRAPH=1
            DO_ONEFORMER3D=1
            shift
            ;;
        --navgraph-voxel-size)
            NAVGRAPH_VOXEL_SIZE="$2"
            shift 2
            ;;
        --navgraph-min-area)
            NAVGRAPH_MIN_AREA="$2"
            shift 2
            ;;
        -*)
            echo "Error: opcion desconocida '$1'"
            echo ""
            usage
            exit 1
            ;;
        *)
            echo "Error: argumento inesperado '$1'"
            echo ""
            usage
            exit 1
            ;;
    esac
done

DATA_ROOT=$(realpath "$DATA_ROOT")
mkdir -p "$DATA_ROOT"

# --- Helper: derive series name from video filename ---
video_to_serie() {
    local filename
    filename=$(basename "$1")
    echo "${filename%.*}"
}

# =============================================
# Mode: --ace-only
# =============================================
if [ -n "$ACE_ONLY" ]; then
    echo "========================================"
    echo " Pipeline: solo ACE"
    echo "========================================"
    echo "Escena ACE   : ${DATA_ROOT}/${ACE_ONLY}"
    echo ""

    ACE_RUN_SH="${SCRIPT_DIR}/../models/ace/docker/run.sh"
    if [ ! -x "$ACE_RUN_SH" ]; then
        echo "Error: no se encontro ${ACE_RUN_SH}"
        exit 1
    fi

    ace_args=(train "$ACE_ONLY")
    if [ -n "$ACE_OUTPUT" ]; then
        ace_args+=("$ACE_OUTPUT")
    else
        ace_args+=("output/$(basename "$ACE_ONLY").pt")
    fi
    ace_args+=(--data-root "$DATA_ROOT")
    [ "$FORCE_CPU" -eq 1 ] && ace_args+=(--cpu)

    "$ACE_RUN_SH" "${ace_args[@]}"
    echo ""
    echo "Pipeline ACE completado."
    exit 0
fi

# =============================================
# Mode: --merge-only
# =============================================
if [ "$MERGE_ONLY" -eq 1 ]; then
    if [ "${#MERGE_SERIES[@]}" -lt 2 ]; then
        echo "Error: --merge-only requiere al menos 2 series."
        exit 1
    fi

    echo "========================================"
    echo " Pipeline: solo merge"
    echo "========================================"
    echo "Series       : ${MERGE_SERIES[*]}"
    echo ""

    MERGE_SH="${SCRIPT_DIR}/merge-colmap.sh"
    merge_args=("${MERGE_SERIES[@]}" --data-root "$DATA_ROOT" --matcher "$MATCHER")
    [ -n "$VOCAB_TREE" ] && merge_args+=(--vocab-tree "$VOCAB_TREE")
    [ "$FORCE_CPU" -eq 1 ] && merge_args+=(--cpu)

    "$MERGE_SH" "${merge_args[@]}"

    if [ "$DO_ACE" -eq 1 ]; then
        echo ""
        echo "Convirtiendo COLMAP merged -> ACE..."
        python3 "${SCRIPT_DIR}/../scripts/colmap2ace.py" \
            --colmap-dir "${DATA_ROOT}/_merged" \
            --output-dir "${DATA_ROOT}/_merged/ace" \
            --train-ratio "$TRAIN_RATIO" \
            --symlink

        ACE_RUN_SH="${SCRIPT_DIR}/../models/ace/docker/run.sh"
        ace_args=(train _merged/ace)
        [ -n "$ACE_OUTPUT" ] && ace_args+=("$ACE_OUTPUT") || ace_args+=("output/_merged.pt")
        ace_args+=(--data-root "$DATA_ROOT")
        [ "$FORCE_CPU" -eq 1 ] && ace_args+=(--cpu)

        "$ACE_RUN_SH" "${ace_args[@]}"
    fi

    if [ "$DO_ONEFORMER3D" -eq 1 ]; then
        echo ""
        echo "Ejecutando OneFormer3D sobre _merged..."
        ONEFORMER_SH="${SCRIPT_DIR}/../run-oneformer3d.sh"
        oneformer_args=(--data-dir "${DATA_ROOT}/_merged")
        [ -n "$ONEFORMER3D_CHECKPOINT" ] && oneformer_args+=(--checkpoint "$ONEFORMER3D_CHECKPOINT")
        [ -n "$ONEFORMER3D_CONFIG" ] && oneformer_args+=(--config "$ONEFORMER3D_CONFIG")
        [ "$FORCE_CPU" -eq 1 ] && oneformer_args+=(--cpu)

        "$ONEFORMER_SH" "${oneformer_args[@]}"
    fi

    if [ "$DO_NAVGRAPH" -eq 1 ]; then
        echo ""
        echo "Generando grafo de navegación..."
        LABELED_PLY="${DATA_ROOT}/_merged/oneformer3d/output/labeled.ply"
        NAVGRAPH_OUTPUT="${DATA_ROOT}/_merged/oneformer3d/navgraph"

        if [ ! -f "$LABELED_PLY" ]; then
            echo "Error: labeled.ply no encontrado en ${LABELED_PLY}"
            exit 1
        fi

        uv run "${SCRIPT_DIR}/../scripts/oneformer3d2navgraph.py" \
            --input "$LABELED_PLY" \
            --output-dir "$NAVGRAPH_OUTPUT" \
            --voxel-size "$NAVGRAPH_VOXEL_SIZE" \
            --min-zone-area "$NAVGRAPH_MIN_AREA"
    fi

    echo ""
    echo "Pipeline merge completado."
    exit 0
fi

# =============================================
# Full pipeline: frames -> COLMAP -> merge -> ACE
# =============================================

echo "========================================"
echo " Pipeline completo"
echo "========================================"

# --- STEP 1: Frame extraction ---
if [ ${#VIDEO_FILES[@]} -gt 0 ]; then
    VIDEOS_DIR=$(realpath "$VIDEOS_DIR")

    if [ ! -d "$VIDEOS_DIR" ]; then
        echo "Error: la carpeta de videos '$VIDEOS_DIR' no existe."
        exit 1
    fi

    if ! command -v videoframeextractor >/dev/null 2>&1; then
        echo "Error: 'videoframeextractor' no encontrado."
        echo "Instalalo con: uv tool install preprocesamiento/VideoFrameExtractor"
        exit 1
    fi

    if ! command -v ffprobe >/dev/null 2>&1; then
        echo "Error: 'ffprobe' no encontrado. Instala ffmpeg."
        exit 1
    fi

    echo ""
    echo "[Paso 1] Extraccion de frames"
    echo "  Videos dir  : $VIDEOS_DIR"
    echo "  Videos      : ${VIDEO_FILES[*]}"
    echo "  Sample FPS  : $SAMPLE_FPS"
    echo ""

    VFE_GPU_FLAG=()
    if [ "$FORCE_CPU" -eq 0 ]; then
        VFE_GPU_FLAG=(--gpu)
    fi

    for video in "${VIDEO_FILES[@]}"; do
        video_path="${VIDEOS_DIR}/${video}"
        if [ ! -f "$video_path" ]; then
            echo "Error: video '$video_path' no encontrado."
            exit 1
        fi

        serie=$(video_to_serie "$video")
        output_dir="${DATA_ROOT}/${serie}/images"
        mkdir -p "$output_dir"

        video_dims=$(ffprobe -v error -select_streams v:0 \
            -show_entries stream=width,height -of csv=s=x:p=0 "$video_path")
        video_w="${video_dims%%x*}"
        video_h="${video_dims##*x}"

        echo "  $video (${video_w}x${video_h}) -> ${serie}/images/"

        videoframeextractor \
            "$video_path" \
            -o "$output_dir" \
            --sample-fps "$SAMPLE_FPS" \
            -w "$video_w" \
            -H "$video_h" \
            -f jpg \
            "${VFE_GPU_FLAG[@]}"

        echo "  Prefijando frames con '${serie}__'..."
        for frame in "${output_dir}/"*; do
            [ -f "$frame" ] || continue
            base=$(basename "$frame")
            if [[ "$base" != "${serie}__"* ]]; then
                mv "$frame" "${output_dir}/${serie}__${base}"
            fi
        done

        SERIES_NAMES+=("$serie")
    done
    echo ""
fi

if [ ${#SERIES_NAMES[@]} -eq 0 ]; then
    echo "Error: no hay series para procesar."
    echo "Usa --series para indicar videos o --series-names para series ya extraidas."
    exit 1
fi

# --- STEP 2: COLMAP parallel ---
echo "[Paso 2] COLMAP paralelo"
echo "  Series      : ${SERIES_NAMES[*]}"
echo "  Max paralelo: $MAX_PARALLEL"
echo "  Modo        : $COLMAP_MODE"
echo ""

PARALLEL_SH="${SCRIPT_DIR}/run-parallel-colmap.sh"
colmap_args=("${SERIES_NAMES[@]}" --max-parallel "$MAX_PARALLEL" --mode "$COLMAP_MODE" --data-root "$DATA_ROOT")
[ "$FORCE_CPU" -eq 1 ] && colmap_args+=(--cpu)

"$PARALLEL_SH" "${colmap_args[@]}"

# --- STEP 3: Merge (optional) ---
if [ "$DO_MERGE" -eq 1 ] && [ "${#SERIES_NAMES[@]}" -ge 2 ]; then
    echo ""
    echo "[Paso 3] Merge de series"

    MERGE_SH="${SCRIPT_DIR}/merge-colmap.sh"
    merge_args=("${SERIES_NAMES[@]}" --data-root "$DATA_ROOT" --matcher "$MATCHER")
    [ -n "$VOCAB_TREE" ] && merge_args+=(--vocab-tree "$VOCAB_TREE")
    [ "$FORCE_CPU" -eq 1 ] && merge_args+=(--cpu)

    "$MERGE_SH" "${merge_args[@]}"
fi

# --- STEP 4: COLMAP -> ACE conversion + training ---
if [ "$DO_ACE" -eq 1 ]; then
    echo ""
    echo "[Paso 4] Conversion COLMAP -> ACE + entrenamiento"

    COLMAP2ACE="${SCRIPT_DIR}/../scripts/colmap2ace.py"
    ACE_RUN_SH="${SCRIPT_DIR}/../models/ace/docker/run.sh"

    if [ "$DO_MERGE" -eq 1 ] && [ "${#SERIES_NAMES[@]}" -ge 2 ]; then
        TARGET="_merged"
    else
        TARGET="${SERIES_NAMES[0]}"
    fi

    echo "  Convirtiendo: ${TARGET} -> ${TARGET}/ace"
    python3 "$COLMAP2ACE" \
        --colmap-dir "${DATA_ROOT}/${TARGET}" \
        --output-dir "${DATA_ROOT}/${TARGET}/ace" \
        --train-ratio "$TRAIN_RATIO" \
        --symlink

    ace_args=(train "${TARGET}/ace")
    if [ -n "$ACE_OUTPUT" ]; then
        ace_args+=("$ACE_OUTPUT")
    else
        ace_args+=("output/${TARGET}.pt")
    fi
    ace_args+=(--data-root "$DATA_ROOT")
    [ "$FORCE_CPU" -eq 1 ] && ace_args+=(--cpu)

    echo "  Entrenando ACE..."
    "$ACE_RUN_SH" "${ace_args[@]}"
fi

# --- STEP 5: OneFormer3D 3D semantic segmentation ---
if [ "$DO_ONEFORMER3D" -eq 1 ]; then
    echo ""
    echo "[Paso 5] Segmentación semántica 3D (OneFormer3D)"

    ONEFORMER_SH="${SCRIPT_DIR}/../run-oneformer3d.sh"
    if [ ! -f "$ONEFORMER_SH" ]; then
        echo "Error: no se encontró ${ONEFORMER_SH}"
        exit 1
    fi

    # Determine target directory (merged or single series)
    if [ "$DO_MERGE" -eq 1 ] && [ "${#SERIES_NAMES[@]}" -ge 2 ]; then
        ONEFORMER_TARGET="${DATA_ROOT}/_merged"
    else
        ONEFORMER_TARGET="${DATA_ROOT}/${SERIES_NAMES[0]}"
    fi

    oneformer_args=(--data-dir "$ONEFORMER_TARGET")
    [ -n "$ONEFORMER3D_CHECKPOINT" ] && oneformer_args+=(--checkpoint "$ONEFORMER3D_CHECKPOINT")
    [ -n "$ONEFORMER3D_CONFIG" ] && oneformer_args+=(--config "$ONEFORMER3D_CONFIG")
    [ "$FORCE_CPU" -eq 1 ] && oneformer_args+=(--cpu)

    echo "  Target: ${ONEFORMER_TARGET}"
    "$ONEFORMER_SH" "${oneformer_args[@]}"
fi

# --- STEP 6: Navigation Graph generation ---
if [ "$DO_NAVGRAPH" -eq 1 ]; then
    echo ""
    echo "[Paso 6] Generación de grafo de navegación"

    # Determine target directory (same as OneFormer3D target)
    if [ "$DO_MERGE" -eq 1 ] && [ "${#SERIES_NAMES[@]}" -ge 2 ]; then
        NAVGRAPH_TARGET="${DATA_ROOT}/_merged"
    else
        NAVGRAPH_TARGET="${DATA_ROOT}/${SERIES_NAMES[0]}"
    fi

    LABELED_PLY="${NAVGRAPH_TARGET}/oneformer3d/output/labeled.ply"
    NAVGRAPH_OUTPUT="${NAVGRAPH_TARGET}/oneformer3d/navgraph"

    if [ ! -f "$LABELED_PLY" ]; then
        echo "Error: labeled.ply no encontrado en ${LABELED_PLY}"
        echo "Asegúrate de que OneFormer3D se ejecutó correctamente."
        exit 1
    fi

    if ! command -v uv >/dev/null 2>&1; then
        echo "Error: 'uv' no encontrado. Instálalo desde https://docs.astral.sh/uv/"
        exit 1
    fi

    echo "  Input : ${LABELED_PLY}"
    echo "  Output: ${NAVGRAPH_OUTPUT}/"

    uv run "${SCRIPT_DIR}/../scripts/oneformer3d2navgraph.py" \
        --input "$LABELED_PLY" \
        --output-dir "$NAVGRAPH_OUTPUT" \
        --voxel-size "$NAVGRAPH_VOXEL_SIZE" \
        --min-zone-area "$NAVGRAPH_MIN_AREA"

    echo ""
    echo "  NavGraph generado en: ${NAVGRAPH_OUTPUT}/"
    echo "  Edita zone_labels.json para asignar nombres semánticos a las zonas."
fi

echo ""
echo "========================================"
echo " Pipeline completado"
echo "========================================"
echo "Series procesadas : ${SERIES_NAMES[*]}"
[ "$DO_MERGE" -eq 1 ] && echo "Merge             : ${DATA_ROOT}/_merged/"
[ "$DO_ACE" -eq 1 ] && echo "Modelo ACE        : ${DATA_ROOT}/output/"
[ "$DO_ONEFORMER3D" -eq 1 ] && echo "OneFormer3D       : segmentación completada"
[ "$DO_NAVGRAPH" -eq 1 ] && echo "NavGraph          : grafo de navegación generado"
echo "========================================"
