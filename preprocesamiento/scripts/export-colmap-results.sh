#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPROCESSING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PREPROCESSING_DIR/data"

if ! command -v zip >/dev/null 2>&1; then
    echo "Error: no se encontro el comando 'zip'."
    echo "Instalalo con tu gestor de paquetes y vuelve a intentarlo."
    exit 1
fi

if [ $# -ge 1 ]; then
    SERIES_NAME="$1"
else
    read -r -p "Nombre de la serie: " SERIES_NAME
fi

if [ -z "$SERIES_NAME" ]; then
    echo "Error: debes indicar el nombre de la serie."
    echo "Uso: $0 <nombre_serie>"
    exit 1
fi

SERIES_DIR="$DATA_DIR/$SERIES_NAME"

if [ ! -d "$SERIES_DIR" ]; then
    echo "Error: la serie '$SERIES_NAME' no existe en '$DATA_DIR'."
    exit 1
fi

ZIP_NAME="${SERIES_NAME}-colmap-results.zip"
ZIP_PATH="$SERIES_DIR/$ZIP_NAME"
TMP_ZIP_PATH="$(mktemp "/tmp/${SERIES_NAME}-colmap-results.XXXXXX.zip")"

cleanup() {
    rm -f "$TMP_ZIP_PATH"
}

trap cleanup EXIT

echo "Serie        : $SERIES_DIR"
echo "Salida zip   : $ZIP_PATH"
echo "Excluyendo   : images/ y archivos de imagen"

cd "$SERIES_DIR"

zip -rq "$TMP_ZIP_PATH" . \
    -x 'images/*' \
    -x '*.zip' \
    -x '*.jpg' \
    -x '*.jpeg' \
    -x '*.png' \
    -x '*.bmp' \
    -x '*.tif' \
    -x '*.tiff' \
    -x '*.webp'

mv -f "$TMP_ZIP_PATH" "$ZIP_PATH"

echo "Zip generado: $ZIP_PATH"
