#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Uso: ./run-cloudcompare.sh <serie> [opciones] [-- archivos_extra]

Asume la estructura:
  preprocesamiento/data/<serie>/

Opciones:
  --data-root PATH   Carpeta base de datasets
  --all              Abre todos los .ply encontrados en la serie
  --mesh             Prioriza la malla antes que la nube fusionada
  -h, --help         Muestra esta ayuda

Comportamiento por defecto:
  1. Abre dense/0/fused.ply si existe
  2. Si no existe, abre meshed-poisson.ply o meshed-delaunay.ply
  3. Si no existe ninguno de los anteriores, abre el primer .ply disponible

Ejemplos:
  ./run-cloudcompare.sh edificio-a
  ./run-cloudcompare.sh edificio-a --mesh
  ./run-cloudcompare.sh edificio-a --all
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${SCRIPT_DIR}/data"
SERIES=""
OPEN_ALL=0
PREFER_MESH=0
EXTRA_FILES=()

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
        --all)
            OPEN_ALL=1
            shift
            ;;
        --mesh)
            PREFER_MESH=1
            shift
            ;;
        --)
            shift
            EXTRA_FILES+=("$@")
            break
            ;;
        -* )
            echo "Error: opcion no reconocida '$1'."
            echo ""
            usage
            exit 1
            ;;
        *)
            if [ -z "$SERIES" ]; then
                SERIES="$1"
            else
                EXTRA_FILES+=("$1")
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

declare -a FILES_TO_OPEN=()

add_if_exists() {
    local path="$1"
    if [ -f "$path" ]; then
        FILES_TO_OPEN+=("$path")
    fi
}

if [ "$OPEN_ALL" -eq 1 ]; then
    while IFS= read -r file; do
        FILES_TO_OPEN+=("$file")
    done < <(find "$SERIES_DIR" -type f \( -name '*.ply' -o -name '*.obj' -o -name '*.stl' \) | sort)
else
    if [ "$PREFER_MESH" -eq 1 ]; then
        add_if_exists "$SERIES_DIR/dense/0/meshed-poisson.ply"
        add_if_exists "$SERIES_DIR/dense/0/meshed-delaunay.ply"
        add_if_exists "$SERIES_DIR/dense/0/fused.ply"
    else
        add_if_exists "$SERIES_DIR/dense/0/fused.ply"
        add_if_exists "$SERIES_DIR/dense/0/meshed-poisson.ply"
        add_if_exists "$SERIES_DIR/dense/0/meshed-delaunay.ply"
    fi

    if [ ${#FILES_TO_OPEN[@]} -eq 0 ]; then
        while IFS= read -r file; do
            FILES_TO_OPEN+=("$file")
            break
        done < <(find "$SERIES_DIR" -type f \( -name '*.ply' -o -name '*.obj' -o -name '*.stl' \) | sort)
    fi
fi

if [ ${#EXTRA_FILES[@]} -gt 0 ]; then
    for candidate in "${EXTRA_FILES[@]}"; do
        if [ -f "$candidate" ]; then
            FILES_TO_OPEN+=("$(realpath "$candidate")")
        elif [ -f "$SERIES_DIR/$candidate" ]; then
            FILES_TO_OPEN+=("$(realpath "$SERIES_DIR/$candidate")")
        else
            echo "Error: no se encontro el archivo '$candidate'."
            exit 1
        fi
    done
fi

if [ ${#FILES_TO_OPEN[@]} -eq 0 ]; then
    echo "Error: no se encontraron resultados visualizables en '$SERIES_DIR'."
    echo "Busca archivos .ply, .obj o .stl dentro de la serie."
    exit 1
fi

deduplicate_files() {
    local -a deduped=()
    local seen=""
    local file
    for file in "$@"; do
        case "|$seen|" in
            *"|$file|"*)
                ;;
            *)
                deduped+=("$file")
                seen+="|$file"
                ;;
        esac
    done
    FILES_TO_OPEN=("${deduped[@]}")
}

deduplicate_files "${FILES_TO_OPEN[@]}"

run_cloudcompare() {
    if command -v CloudCompare >/dev/null 2>&1; then
        exec CloudCompare "$@"
    fi

    if command -v ccViewer >/dev/null 2>&1; then
        exec ccViewer "$@"
    fi

    if command -v flatpak >/dev/null 2>&1 && flatpak info org.cloudcompare.CloudCompare >/dev/null 2>&1; then
        exec flatpak run org.cloudcompare.CloudCompare "$@"
    fi

    echo "Error: no se encontro CloudCompare en el sistema."
    echo "Instalalo con ./scripts/install_tools.sh o usa flatpak/apt segun tu entorno."
    exit 1
}

echo "Serie        : $SERIES_DIR"
printf 'Abriendo     : %s\n' "${FILES_TO_OPEN[@]}"

run_cloudcompare "${FILES_TO_OPEN[@]}"
