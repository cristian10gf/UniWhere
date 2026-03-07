#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VFE_PKG="${SCRIPT_DIR}/VideoFrameExtractor"
VIDEOS_DIR="${SCRIPT_DIR}/videos"
DATA_ROOT="${SCRIPT_DIR}/data"

usage() {
    cat <<'EOF'
Uso: ./run-extract-frames.sh VIDEO [opciones]

Extrae frames de un video y los coloca en la estructura esperada por COLMAP.
El video debe estar en la carpeta preprocesamiento/videos/.

Argumentos:
  VIDEO                Nombre del archivo de video (relativo a videos/)

Opciones:
  --sample-fps FPS     FPS de muestreo (default: 2)
  --gpu                Usar GPU para decodificación
  --data-root PATH     Carpeta base de datos (default: preprocesamiento/data)
  -h, --help           Mostrar esta ayuda

Ejemplo:
  ./run-extract-frames.sh campus-norte.mp4 --sample-fps 3

Resultado:
  data/campus-norte/images/frame000000.jpg
  data/campus-norte/images/frame000001.jpg
  ...

Luego puedes ejecutar COLMAP con:
  ./run-colmap.sh campus-norte
EOF
}

VIDEO=""
SAMPLE_FPS=2
GPU_FLAGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --sample-fps) SAMPLE_FPS="$2"; shift 2 ;;
        --gpu) GPU_FLAGS=(--gpu); shift ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        -*)
            echo "Error: opción desconocida '$1'"
            echo ""
            usage
            exit 1
            ;;
        *)
            if [ -z "$VIDEO" ]; then
                VIDEO="$1"
            else
                echo "Error: argumento inesperado '$1'"
                echo ""
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$VIDEO" ]; then
    echo "Error: se requiere el nombre del video."
    echo ""
    usage
    exit 1
fi

video_path="${VIDEOS_DIR}/${VIDEO}"
if [ ! -f "$video_path" ]; then
    echo "Error: video '${video_path}' no encontrado."
    echo "Asegúrate de que el video esté en: ${VIDEOS_DIR}/"
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: 'uv' no encontrado. Instálalo desde https://docs.astral.sh/uv/"
    exit 1
fi

serie="${VIDEO%.*}"
output_dir="${DATA_ROOT}/${serie}/images"

echo "Obteniendo información del video..."
if ! info_output=$(videoframeextractor "$video_path" --info 2>&1); then
    echo "Error: no se pudo obtener información del video."
    echo "$info_output"
    exit 1
fi

resolution=$(echo "$info_output" | grep -i "resol" | grep -oE '[0-9]+x[0-9]+' | head -1)
if [ -z "$resolution" ]; then
    echo "Error: no se pudo determinar la resolución del video."
    echo "Salida de videoframeextractor:"
    echo "$info_output"
    exit 1
fi

video_w="${resolution%%x*}"
video_h="${resolution##*x}"

echo ""
echo "========================================"
echo " Extracción de frames"
echo "========================================"
echo "  Video       : ${VIDEO} (${video_w}x${video_h})"
echo "  Serie       : ${serie}"
echo "  Sample FPS  : ${SAMPLE_FPS}"
echo "  Salida      : ${output_dir}"
echo "========================================"
echo ""

mkdir -p "$output_dir"

videoframeextractor \
    "$video_path" \
    -o "$output_dir" \
    --sample-fps "$SAMPLE_FPS" \
    -w "$video_w" \
    -H "$video_h" \
    -f jpg \
    ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"}

echo ""
echo "Frames extraídos en: ${output_dir}"
echo "Para ejecutar COLMAP: ./run-colmap.sh ${serie}"
