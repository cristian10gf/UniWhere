#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Uso: ./run-parallel-mast3r.sh <serie1> [serie2 ...] [opciones] [-- args_extra_mast3r]

Ejecuta MASt3R en paralelo sobre multiples series con control de concurrencia.
Cada serie se procesa via run-mast3r.sh.

Opciones:
  --max-parallel N    Maximo de ejecuciones simultaneas (default: 1)
  --mode MODE         Modo MASt3R: automatic|advanced (default: advanced)
  --data-root PATH    Carpeta base de datos (default: preprocesamiento/data)
  --matcher TYPE      sequential|exhaustive|vocab_tree (default: depende de --mode)
  --cpu               Forzar modo CPU
  -h, --help          Mostrar esta ayuda

Ejemplos:
  ./run-parallel-mast3r.sh serie-1 serie-2 serie-3
  ./run-parallel-mast3r.sh serie-1 serie-2 --max-parallel 2 --matcher exhaustive
  ./run-parallel-mast3r.sh serie-1 serie-2 -- --conf-thr 1.2
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
RUN_SERIES_SH="${SCRIPT_DIR}/../run-mast3r.sh"
DATA_ROOT="${PROJECT_ROOT}/preprocesamiento/data"
MAX_PARALLEL=1
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
        --max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
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

if [ "${#SERIES[@]}" -eq 0 ]; then
    echo "Error: debes indicar al menos una serie."
    echo ""
    usage
    exit 1
fi

if ! [ "$MAX_PARALLEL" -gt 0 ] 2>/dev/null; then
    echo "Error: --max-parallel debe ser un entero positivo."
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

if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: la carpeta de datos '$DATA_ROOT' no existe."
    exit 1
fi

if [ ! -x "$RUN_SERIES_SH" ]; then
    echo "Error: no se encontro run-mast3r.sh en '${RUN_SERIES_SH}'."
    exit 1
fi

for serie in "${SERIES[@]}"; do
    if [ ! -d "${DATA_ROOT}/${serie}/images" ]; then
        echo "Error: '${DATA_ROOT}/${serie}/images' no existe."
        echo "Cada serie debe tener su carpeta de imagenes."
        exit 1
    fi
done

RESULTS_DIR=$(mktemp -d)
trap 'rm -rf "$RESULTS_DIR"' EXIT

run_mast3r_for_serie() {
    local serie="$1"
    local log="${DATA_ROOT}/${serie}/mast3r.log"

    local args=("$serie" --mode "$MODE" --data-root "$DATA_ROOT" --matcher "$MATCHER")

    if [ "$FORCE_CPU" -eq 1 ]; then
        args+=(--cpu)
    fi

    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        args+=(--)
        args+=("${EXTRA_ARGS[@]}")
    fi

    echo "[$(date +%H:%M:%S)] Iniciando MASt3R: $serie (log: $log)"

    if "$RUN_SERIES_SH" "${args[@]}" > "$log" 2>&1; then
        echo "OK" > "${RESULTS_DIR}/${serie}"
        echo "[$(date +%H:%M:%S)] OK: $serie"
    else
        echo "FAIL" > "${RESULTS_DIR}/${serie}"
        echo "[$(date +%H:%M:%S)] FALLO: $serie (ver $log)"
    fi
}

echo "========================================"
echo " MASt3R paralelo"
echo "========================================"
echo "Series       : ${SERIES[*]}"
echo "Max paralelo : $MAX_PARALLEL"
echo "Modo         : $MODE"
echo "Matcher      : $MATCHER"
echo "CPU          : $([ "$FORCE_CPU" -eq 1 ] && echo "si" || echo "no")"
echo "Data root    : $DATA_ROOT"
echo "========================================"
echo ""

active=0
for serie in "${SERIES[@]}"; do
    if [ "$active" -ge "$MAX_PARALLEL" ]; then
        wait -n 2>/dev/null || true
        ((active--))
    fi

    run_mast3r_for_serie "$serie" &
    ((active++))
done

wait

successes=0
failures=0
failed_series=()

for serie in "${SERIES[@]}"; do
    result=$(cat "${RESULTS_DIR}/${serie}" 2>/dev/null || echo "UNKNOWN")
    if [ "$result" = "OK" ]; then
        ((successes++))
    else
        ((failures++))
        failed_series+=("$serie")
    fi
done

echo ""
echo "========================================"
echo " Resumen"
echo "========================================"
echo "Total series : ${#SERIES[@]}"
echo "Exitosas     : $successes"
echo "Fallidas     : $failures"

if [ ${#failed_series[@]} -gt 0 ]; then
    echo "Series con error:"
    for serie in "${failed_series[@]}"; do
        echo "  - $serie (ver ${DATA_ROOT}/${serie}/mast3r.log)"
    done
    echo ""
    echo "Fallback recomendado: reintenta con --reconstructor colmap para esas series."
fi

echo "========================================"

[ "$failures" -eq 0 ]
