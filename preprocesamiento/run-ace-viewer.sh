#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Uso: ./run-ace-viewer.sh <serie> [opciones]

Visualizador interactivo de resultados ACE usando Rerun.
Carga la nube de puntos de COLMAP y las predicciones de ACE.

Asume la estructura:
  preprocesamiento/data/<serie>/         (directorio COLMAP con dense/ o sparse/)
  preprocesamiento/data/<serie>/ace/     (dataset ACE con train/ y test/)
  preprocesamiento/data/output/<serie>.pt (modelo ACE entrenado, opcional)

Opciones:
  --data-root PATH       Carpeta base de datasets (default: preprocesamiento/data)
  --scene PATH           Path al dataset ACE (default: <serie>/ace)
  --point-cloud FILE     PLY especifico en vez de auto-detectar
  --test-poses FILE      Archivo de resultados ACE (poses_*.txt)
  --model FILE           Head entrenado (.pt) para nube ACE adicional
  --max-points N         Maximo de puntos (default: 1000000)
  --export-ply FILE      Exportar nube a PLY para CloudCompare
  -h, --help             Muestra esta ayuda

Ejemplos:
  ./run-ace-viewer.sh _merged
  ./run-ace-viewer.sh _merged --test-poses data/output/poses__merged.txt
  ./run-ace-viewer.sh serie-1 --point-cloud data/serie-1/dense/0/fused.ply
  ./run-ace-viewer.sh _merged --export-ply nube.ply
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIEWER_DIR="${SCRIPT_DIR}/visualizadores/ace-rerun"
DATA_ROOT="${SCRIPT_DIR}/data"
SERIES=""
SCENE_OVERRIDE=""
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
        --scene)
            SCENE_OVERRIDE="$2"
            shift 2
            ;;
        --point-cloud|--test-poses|--model|--encoder|--export-ply)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --max-points|--filter-depth|--image-height)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --ace-dir)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        -*)
            echo "Error: opcion no reconocida '$1'."
            echo ""
            usage
            exit 1
            ;;
        *)
            if [ -z "$SERIES" ]; then
                SERIES="$1"
            else
                echo "Error: argumento inesperado '$1'."
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$SERIES" ]; then
    echo "Error: debes indicar el nombre de la serie."
    echo ""
    usage
    exit 1
fi

DATA_ROOT=$(realpath "$DATA_ROOT")
SERIES_DIR="${DATA_ROOT}/${SERIES}"

if [ ! -d "$SERIES_DIR" ]; then
    echo "Error: la serie '$SERIES' no existe en '$DATA_ROOT'."
    exit 1
fi

if [ -n "$SCENE_OVERRIDE" ]; then
    SCENE_PATH="$SCENE_OVERRIDE"
else
    SCENE_PATH="${SERIES_DIR}/ace"
fi

if [ ! -d "$SCENE_PATH" ]; then
    echo "Error: dataset ACE no encontrado en '$SCENE_PATH'."
    echo "Ejecuta primero colmap2ace.py para convertir los datos."
    exit 1
fi

HAS_PC=0
for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
    if [ "$arg" = "--point-cloud" ]; then
        HAS_PC=1
        break
    fi
done

COLMAP_ARGS=()
if [ "$HAS_PC" -eq 0 ]; then
    COLMAP_ARGS+=("--colmap-dir" "$SERIES_DIR")
fi

UV_BIN="${HOME}/.local/bin/uv"
if ! command -v uv >/dev/null 2>&1; then
    if [ -x "$UV_BIN" ]; then
        PATH="${HOME}/.local/bin:$PATH"
    else
        echo "Error: uv no encontrado. Instalalo con: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
fi

echo "Serie        : $SERIES"
echo "COLMAP dir   : $SERIES_DIR"
echo "ACE scene    : $SCENE_PATH"
echo ""

cd "$VIEWER_DIR"
exec uv run visualize_ace.py \
    --scene "$SCENE_PATH" \
    "${COLMAP_ARGS[@]+"${COLMAP_ARGS[@]}"}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
