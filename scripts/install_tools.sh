#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VFE_DIR="$REPO_ROOT/preprocesamiento/VideoFrameExtractor"
TOOLS_DIR="$REPO_ROOT/.tools"
VENV_DIR="$TOOLS_DIR/videoframeextractor"
VFE_TOOL_NAME="videoframeextractor"

log() {
    printf '[INFO] %s\n' "$1"
}

fail() {
    printf '[ERROR] %s\n' "$1" >&2
    exit 1
}

require_python312() {
    local candidate
    for candidate in python3.12 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
                printf '%s' "$candidate"
                return 0
            fi
        fi
    done

    return 1
}

install_videoframeextractor() {
    if command -v uv >/dev/null 2>&1; then
        log "Instalando VideoFrameExtractor como tool de uv desde $VFE_DIR"
        uv tool install --editable --force "$VFE_DIR"
        log "VideoFrameExtractor instalado con uv. Ejecutable esperado en PATH: $VFE_TOOL_NAME"
        log "Si el comando no aparece todavía en la terminal actual, abre una nueva sesión o ejecuta: uv tool update-shell"
    else
        local python_cmd
        python_cmd="$(require_python312)" || fail "Se requiere Python 3.12+ para instalar VideoFrameExtractor"

        mkdir -p "$TOOLS_DIR"
        log "Instalando VideoFrameExtractor con $python_cmd en $VENV_DIR"
        "$python_cmd" -m venv "$VENV_DIR"
        "$VENV_DIR/bin/python" -m pip install --upgrade pip
        "$VENV_DIR/bin/python" -m pip install --editable "$VFE_DIR"
        log "VideoFrameExtractor instalado. Ejecutable esperado: $VENV_DIR/bin/$VFE_TOOL_NAME"
    fi
}

ensure_flathub_remote() {
    if ! flatpak remote-list | awk '{print $1}' | grep -qx flathub; then
        log "Agregando remoto flathub para instalar CloudCompare"
        flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
    fi
}

install_cloudcompare() {
    if command -v flatpak >/dev/null 2>&1; then
        log "Instalando CloudCompare con flatpak"
        ensure_flathub_remote
        flatpak install -y flathub org.cloudcompare.CloudCompare
        log "CloudCompare instalado. Ejecuta: flatpak run org.cloudcompare.CloudCompare"
        return 0
    fi

    if command -v apt-get >/dev/null 2>&1; then
        log "flatpak no está disponible. Intentando instalar CloudCompare con apt-get"
        sudo apt-get update
        sudo apt-get install -y cloudcompare
        log "CloudCompare instalado mediante apt-get"
        return 0
    fi

    fail "No se encontró un método automático para instalar CloudCompare. Instala flatpak o usa una distribución con apt-get"
}

main() {
    [[ -d "$VFE_DIR" ]] || fail "No se encontró el directorio de VideoFrameExtractor en $VFE_DIR"

    install_videoframeextractor
    install_cloudcompare

    log "Instalación completada"
}

main "$@"