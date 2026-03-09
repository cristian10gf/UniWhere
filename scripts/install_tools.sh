#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VFE_DIR="$REPO_ROOT/preprocesamiento/VideoFrameExtractor"
TOOLS_DIR="$REPO_ROOT/.tools"
VENV_DIR="$TOOLS_DIR/videoframeextractor"
VFE_TOOL_NAME="videoframeextractor"
COLMAP_DIR="$REPO_ROOT/preprocesamiento/models/colmap"
ACE_DIR="$REPO_ROOT/preprocesamiento/models/ace"

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

install_ffmpeg() {
    if command -v ffmpeg >/dev/null 2>&1; then
        log "ffmpeg ya está instalado: $(ffmpeg -version 2>&1 | head -1)"
        return 0
    fi

    if command -v apt-get >/dev/null 2>&1; then
        log "Instalando ffmpeg con apt-get"
        sudo apt-get update
        sudo apt-get install -y ffmpeg
    elif command -v brew >/dev/null 2>&1; then
        log "Instalando ffmpeg con brew"
        brew install ffmpeg
    elif command -v dnf >/dev/null 2>&1; then
        log "Instalando ffmpeg con dnf"
        sudo dnf install -y ffmpeg
    elif command -v pacman >/dev/null 2>&1; then
        log "Instalando ffmpeg con pacman"
        sudo pacman -S --noconfirm ffmpeg
    else
        fail "No se encontró un gestor de paquetes para instalar ffmpeg. Instálalo manualmente."
    fi

    log "ffmpeg instalado: $(ffmpeg -version 2>&1 | head -1)"
}

build_colmap_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        log "Docker no está instalado. Omitiendo imagen COLMAP."
        return 0
    fi

    if docker image inspect colmap:latest >/dev/null 2>&1; then
        log "Imagen colmap:latest ya existe localmente"
        log "Para reconstruir con soporte GPU (Ceres + cuSOLVER): $0 --rebuild-colmap"
        return 0
    fi

    if [[ ! -f "$COLMAP_DIR/docker/Dockerfile" ]]; then
        log "Advertencia: no se encontró Dockerfile de COLMAP en $COLMAP_DIR/docker/."
        log "Descargando imagen oficial (sin GPU bundle adjustment)..."
        docker pull colmap/colmap:latest || true
        return 0
    fi

    log "Construyendo imagen colmap:latest desde fuente (Ceres con CUDA/cuSOLVER)..."
    log "Esto toma ~20-30 minutos la primera vez."
    docker build -f "$COLMAP_DIR/docker/Dockerfile" -t colmap:latest "$COLMAP_DIR"
    log "Imagen colmap:latest construida con soporte GPU para bundle adjustment"
}

rebuild_colmap_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        fail "Docker no está instalado."
    fi

    if [[ ! -f "$COLMAP_DIR/docker/Dockerfile" ]]; then
        fail "No se encontró Dockerfile de COLMAP en $COLMAP_DIR/docker/"
    fi

    log "Reconstruyendo imagen colmap:latest desde fuente (Ceres con CUDA/cuSOLVER)..."
    docker build --no-cache -f "$COLMAP_DIR/docker/Dockerfile" -t colmap:latest "$COLMAP_DIR"
    log "Imagen colmap:latest reconstruida con soporte GPU para bundle adjustment"
}

build_ace_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        log "Docker no está instalado. Omitiendo imagen ACE."
        return 0
    fi

    if docker image inspect ace:latest >/dev/null 2>&1; then
        log "Imagen ace:latest ya existe localmente"
        return 0
    fi

    if [[ ! -f "$ACE_DIR/docker/Dockerfile" ]]; then
        log "Advertencia: no se encontró Dockerfile de ACE en $ACE_DIR/docker/. Omitiendo."
        return 0
    fi

    local encoder="$ACE_DIR/ace_encoder_pretrained.pt"
    if [[ ! -f "$encoder" ]]; then
        log "Descargando pesos del encoder ACE preentrenado..."
        if command -v wget >/dev/null 2>&1; then
            wget -q --show-progress -O "$ACE_DIR/ace_models.tar.gz" \
                "https://storage.googleapis.com/niantic-lon-static/research/ace/ace_models.tar.gz"
        elif command -v curl >/dev/null 2>&1; then
            curl -L -o "$ACE_DIR/ace_models.tar.gz" \
                "https://storage.googleapis.com/niantic-lon-static/research/ace/ace_models.tar.gz"
        else
            log "Advertencia: no se encontró wget ni curl. Descarga ace_encoder_pretrained.pt manualmente."
            log "  URL: https://storage.googleapis.com/niantic-lon-static/research/ace/ace_models.tar.gz"
            log "  Destino: $encoder"
            return 0
        fi
        tar -xzf "$ACE_DIR/ace_models.tar.gz" -C "$ACE_DIR" ace_encoder_pretrained.pt 2>/dev/null || \
            tar -xzf "$ACE_DIR/ace_models.tar.gz" -C "$ACE_DIR" 2>/dev/null
        rm -f "$ACE_DIR/ace_models.tar.gz"
    fi

    if [[ -f "$encoder" ]]; then
        log "Construyendo imagen ace:latest desde $ACE_DIR"
        docker build -f "$ACE_DIR/docker/Dockerfile" -t ace:latest "$ACE_DIR"
        log "Imagen ace:latest construida"
    else
        log "Advertencia: ace_encoder_pretrained.pt no encontrado. La imagen se construirá sin él."
        log "  El script run.sh intentará construir la imagen al ejecutarse."
    fi
}

main() {
    [[ -d "$VFE_DIR" ]] || fail "No se encontró el directorio de VideoFrameExtractor en $VFE_DIR"

    for arg in "$@"; do
        case "$arg" in
            --rebuild-colmap)
                rebuild_colmap_docker
                log "Reconstrucción de COLMAP completada"
                return 0
                ;;
        esac
    done

    install_ffmpeg
    install_videoframeextractor
    install_cloudcompare
    build_colmap_docker
    build_ace_docker

    log "Instalación completada"
}

main "$@"