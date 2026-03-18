#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Uso: ./merge-mast3r.sh <serie1> <serie2> [serie3 ...] [opciones] [-- args_extra_mast3r]

Genera una reconstruccion unificada con MASt3R para multiples series sin usar
el merge hibrido de COLMAP. El script:
  1. Prepara data/<output>/images con links/copias de todas las series
  2. Ejecuta MASt3R sobre el conjunto combinado
  3. Deja artefactos compatibles en data/<output>/

Artefactos esperados:
  - data/<output>/database.db
  - data/<output>/sparse/0/{cameras,images,points3D}.{txt|bin}
  - data/<output>/dense/0/fused.ply

Opciones:
  --data-root PATH      Carpeta base de datos (default: preprocesamiento/data)
  --output-name NAME    Nombre de salida (default: _merged)
  --mode MODE           Modo MASt3R: automatic|advanced (default: advanced)
  --matcher TYPE        sequential|exhaustive|vocab_tree (default: depende de --mode)
  --cpu                 Forzar modo CPU
  -h, --help            Mostrar esta ayuda

Ejemplos:
  ./merge-mast3r.sh serie-1 serie-2 serie-3
  ./merge-mast3r.sh serie-1 serie-2 --output-name escena-a
  ./merge-mast3r.sh serie-1 serie-2 --matcher sequential -- --overlap 25
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
RUN_MAST3R_SH="${SCRIPT_DIR}/../run-mast3r.sh"
DATA_ROOT="${PROJECT_ROOT}/preprocesamiento/data"
OUTPUT_NAME="_merged"
MODE="advanced"
MATCHER=""
FORCE_CPU=0
SERIES=()
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
        --output-name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --matcher)
            MATCHER="$2"
            shift 2
            ;;
        --cpu)
            FORCE_CPU=1
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

case "$MODE" in
    automatic|advanced) ;;
    *)
        echo "Error: --mode invalido '$MODE'. Usa automatic o advanced."
        exit 1
        ;;
esac

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
        echo "Error: --matcher invalido '$MATCHER'."
        exit 1
        ;;
esac

DATA_ROOT=$(realpath "$DATA_ROOT")
MERGED_DIR="${DATA_ROOT}/${OUTPUT_NAME}"
MERGED_IMAGES_DIR="${MERGED_DIR}/images"
LOG_PATH="${MERGED_DIR}/mast3r-merge.log"

if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: la carpeta de datos '$DATA_ROOT' no existe."
    exit 1
fi

if [ ! -x "$RUN_MAST3R_SH" ]; then
    echo "Error: no se encontro run-mast3r.sh en '${RUN_MAST3R_SH}'."
    exit 1
fi

for serie in "${SERIES[@]}"; do
    if [ "$serie" = "$OUTPUT_NAME" ]; then
        echo "Error: una serie de entrada coincide con --output-name ('$OUTPUT_NAME')."
        exit 1
    fi

    if [ ! -d "${DATA_ROOT}/${serie}/images" ]; then
        echo "Error: '${DATA_ROOT}/${serie}/images' no existe."
        exit 1
    fi
done

link_or_copy() {
    local src="$1"
    local dst="$2"

    if ln -s "$src" "$dst" 2>/dev/null; then
        return 0
    fi

    cp "$src" "$dst"
}

echo "========================================"
echo " Merge MASt3R"
echo "========================================"
echo "Series       : ${SERIES[*]}"
echo "Salida       : ${MERGED_DIR}"
echo "Modo         : $MODE"
echo "Matcher      : $MATCHER"
echo "CPU          : $([ "$FORCE_CPU" -eq 1 ] && echo "si" || echo "no")"
echo "========================================"
echo ""

echo "[1/2] Preparando imagenes combinadas..."
mkdir -p "$MERGED_IMAGES_DIR"

if [ -d "$MERGED_IMAGES_DIR" ]; then
    find "$MERGED_IMAGES_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
fi

added=0
for serie in "${SERIES[@]}"; do
    while IFS= read -r -d '' img; do
        base=$(basename "$img")
        target="${MERGED_IMAGES_DIR}/${base}"

        if [ -e "$target" ] || [ -L "$target" ]; then
            echo "Error: nombre de imagen duplicado al mergear: $base"
            echo "Asegura prefijos unicos por serie antes del merge."
            exit 1
        fi

        link_or_copy "$(realpath "$img")" "$target"
        ((added++))
    done < <(find "${DATA_ROOT}/${serie}/images" -maxdepth 1 -type f -print0)
done

echo "  Imagenes preparadas: ${added}"
echo ""

echo "[2/2] Ejecutando reconstruccion MASt3R para ${OUTPUT_NAME}..."

run_args=("$OUTPUT_NAME" --data-root "$DATA_ROOT" --mode "$MODE" --matcher "$MATCHER")
[ "$FORCE_CPU" -eq 1 ] && run_args+=(--cpu)

if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    run_args+=(--)
    run_args+=("${EXTRA_ARGS[@]}")
fi

if "$RUN_MAST3R_SH" "${run_args[@]}" > "$LOG_PATH" 2>&1; then
    echo "  MASt3R merge completado (log: ${LOG_PATH})"
else
    echo "Error: MASt3R merge fallo. Revisa: ${LOG_PATH}"
    echo "Fallback recomendado: rerun con --reconstructor colmap."
    exit 1
fi

if [ ! -f "${MERGED_DIR}/database.db" ]; then
    echo "Error: no se genero ${MERGED_DIR}/database.db"
    exit 1
fi

if [ ! -f "${MERGED_DIR}/dense/0/fused.ply" ]; then
    echo "Error: no se genero ${MERGED_DIR}/dense/0/fused.ply"
    exit 1
fi

if [ ! -d "${MERGED_DIR}/sparse/0" ]; then
    echo "Error: no se genero ${MERGED_DIR}/sparse/0"
    exit 1
fi

echo ""
echo "========================================"
echo " Merge MASt3R completado"
echo "========================================"
echo "database.db     : ${MERGED_DIR}/database.db"
echo "images          : ${MERGED_IMAGES_DIR}/"
echo "sparse model    : ${MERGED_DIR}/sparse/0/"
echo "dense cloud     : ${MERGED_DIR}/dense/0/fused.ply"
echo "========================================"
