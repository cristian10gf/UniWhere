param(
    [switch]$RebuildColmap
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$VfeDir = Join-Path $RepoRoot 'preprocesamiento/VideoFrameExtractor'
$ToolsDir = Join-Path $RepoRoot '.tools'
$VenvDir = Join-Path $ToolsDir 'videoframeextractor'
$VfeToolName = 'videoframeextractor'
$ColmapDir = Join-Path $RepoRoot 'preprocesamiento/models/colmap'
$AceDir = Join-Path $RepoRoot 'preprocesamiento/models/ace'
$Mast3rSubmoduleDir = Join-Path $RepoRoot 'preprocesamiento/models/mast3r'
$Mast3rIntegrationDir = Join-Path $RepoRoot 'preprocesamiento/models/mast3r-integration'
$Mast3rWeightsDir = Join-Path $RepoRoot 'preprocesamiento/data/weights/mast3r'
$Mast3rMainWeightName = 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
$Mast3rMainWeightUrl = 'https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message"
}

function Get-PythonCommand {
    $candidates = @(
        @{ Command = 'py'; Args = @('-3.12') },
        @{ Command = 'python'; Args = @() }
    )

    foreach ($candidate in $candidates) {
        if (Get-Command $candidate.Command -ErrorAction SilentlyContinue) {
            try {
                & $candidate.Command @($candidate.Args + @('-c', 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'))
                if ($LASTEXITCODE -eq 0) {
                    return $candidate
                }
            }
            catch {
            }
        }
    }

    throw 'Se requiere Python 3.12 o superior para instalar VideoFrameExtractor.'
}

function Install-VideoFrameExtractor {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Write-Info "Instalando VideoFrameExtractor como tool de uv desde $VfeDir"
        uv tool install --editable --force $VfeDir
        Write-Info "VideoFrameExtractor instalado con uv. Ejecutable esperado en PATH: $VfeToolName"
        Write-Info 'Si el comando no aparece todavía en la sesión actual, abre una nueva terminal o ejecuta: uv tool update-shell'
    }
    else {
        $python = Get-PythonCommand
        New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
        Write-Info "Instalando VideoFrameExtractor con $($python.Command) en $VenvDir"
        & $python.Command @($python.Args + @('-m', 'venv', $VenvDir))
        & (Join-Path $VenvDir 'Scripts/python.exe') -m pip install --upgrade pip
        & (Join-Path $VenvDir 'Scripts/python.exe') -m pip install --editable $VfeDir
        Write-Info "VideoFrameExtractor instalado. Ejecutable esperado: $(Join-Path $VenvDir 'Scripts/videoframeextractor.exe')"
    }
}

function Install-CloudCompare {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Info 'Instalando CloudCompare con winget'
        winget install --exact --id CloudCompare.CloudCompare --accept-package-agreements --accept-source-agreements
        Write-Info 'CloudCompare instalado con winget'
        return
    }

    throw 'No se encontró winget para instalar CloudCompare automáticamente en Windows.'
}

function Install-Ffmpeg {
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
        Write-Info "ffmpeg ya está instalado"
        return
    }

    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Info 'Instalando ffmpeg con winget'
        winget install --exact --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
        Write-Info 'ffmpeg instalado con winget. Reinicia la terminal para que esté en PATH.'
        return
    }

    if (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Info 'Instalando ffmpeg con chocolatey'
        choco install ffmpeg -y
        Write-Info 'ffmpeg instalado con chocolatey'
        return
    }

    Write-Info 'Advertencia: no se pudo instalar ffmpeg automáticamente. Instálalo manualmente desde https://ffmpeg.org/download.html'
}

function Build-ColmapDocker {
    param([switch]$Rebuild)

    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Info 'Docker no está instalado. Omitiendo imagen COLMAP.'
        return
    }

    $dockerfile = Join-Path $ColmapDir 'docker/Dockerfile'

    if (-not $Rebuild) {
        $inspect = docker image inspect colmap:latest 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Info 'Imagen colmap:latest ya existe localmente'
            Write-Info 'Para reconstruir con soporte GPU (Ceres + cuSOLVER): .\install_tools.ps1 -RebuildColmap'
            return
        }
    }

    if (-not (Test-Path $dockerfile)) {
        Write-Info 'Advertencia: no se encontró Dockerfile de COLMAP. Descargando imagen oficial (sin GPU bundle adjustment)...'
        docker pull colmap/colmap:latest
        return
    }

    Write-Info 'Construyendo imagen colmap:latest desde fuente (Ceres con CUDA/cuSOLVER)...'
    Write-Info 'Esto toma ~20-30 minutos la primera vez.'
    if ($Rebuild) {
        docker build --no-cache -f $dockerfile -t colmap:latest $ColmapDir
    } else {
        docker build -f $dockerfile -t colmap:latest $ColmapDir
    }
    if ($LASTEXITCODE -eq 0) {
        Write-Info 'Imagen colmap:latest construida con soporte GPU para bundle adjustment'
    } else {
        throw 'Error al construir la imagen de COLMAP.'
    }
}

function Build-AceDocker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Info 'Docker no está instalado. Omitiendo imagen ACE.'
        return
    }

    $inspect = docker image inspect ace:latest 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Info 'Imagen ace:latest ya existe localmente'
        return
    }

    $dockerfile = Join-Path $AceDir 'docker/Dockerfile'
    if (-not (Test-Path $dockerfile)) {
        Write-Info "Advertencia: no se encontró Dockerfile de ACE en $dockerfile. Omitiendo."
        return
    }

    $encoder = Join-Path $AceDir 'ace_encoder_pretrained.pt'
    if (-not (Test-Path $encoder)) {
        Write-Info 'Descargando pesos del encoder ACE preentrenado...'
        $tarball = Join-Path $AceDir 'ace_models.tar.gz'
        try {
            Invoke-WebRequest -Uri 'https://storage.googleapis.com/niantic-lon-static/research/ace/ace_models.tar.gz' -OutFile $tarball
            tar -xzf $tarball -C $AceDir
            Remove-Item $tarball -ErrorAction SilentlyContinue
        } catch {
            Write-Info "Advertencia: no se pudieron descargar los pesos. Descarga ace_encoder_pretrained.pt manualmente."
            Write-Info "  URL: https://storage.googleapis.com/niantic-lon-static/research/ace/ace_models.tar.gz"
            return
        }
    }

    if (Test-Path $encoder) {
        Write-Info "Construyendo imagen ace:latest desde $AceDir"
        docker build -f $dockerfile -t ace:latest $AceDir
        Write-Info 'Imagen ace:latest construida'
    } else {
        Write-Info 'Advertencia: ace_encoder_pretrained.pt no encontrado. La imagen se construirá al ejecutar run.sh.'
    }
}

function Install-Mast3rWeights {
    New-Item -ItemType Directory -Force -Path $Mast3rWeightsDir | Out-Null
    $weightPath = Join-Path $Mast3rWeightsDir $Mast3rMainWeightName

    if (Test-Path $weightPath) {
        Write-Info "Peso MASt3R ya disponible: $weightPath"
        return
    }

    Write-Info "Descargando peso MASt3R principal en $weightPath"
    try {
        Invoke-WebRequest -Uri $Mast3rMainWeightUrl -OutFile $weightPath
        Write-Info "Peso MASt3R descargado: $weightPath"
    }
    catch {
        Write-Info 'Advertencia: no se pudo descargar el peso MASt3R automáticamente.'
        Write-Info "  URL: $Mast3rMainWeightUrl"
        Write-Info "  Destino esperado: $weightPath"
    }
}

function Build-Mast3rDocker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Info 'Docker no está instalado. Omitiendo imagen MASt3R.'
        return
    }

    $inspect = docker image inspect mast3r-pipeline:latest 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Info 'Imagen mast3r-pipeline:latest ya existe localmente'
        return
    }

    $dockerfile = Join-Path $Mast3rIntegrationDir 'docker/Dockerfile'
    if (-not (Test-Path $dockerfile)) {
        Write-Info "Advertencia: no se encontró Dockerfile de MASt3R en $dockerfile. Omitiendo."
        return
    }

    if (-not (Test-Path $Mast3rSubmoduleDir)) {
        Write-Info "Advertencia: submódulo MASt3R no encontrado en $Mast3rSubmoduleDir"
        Write-Info '  Ejecuta: git submodule update --init --recursive'
        return
    }

    Write-Info "Construyendo imagen mast3r-pipeline:latest desde $dockerfile"
    docker build -f $dockerfile -t mast3r-pipeline:latest $RepoRoot
    if ($LASTEXITCODE -eq 0) {
        Write-Info 'Imagen mast3r-pipeline:latest construida'
    }
    else {
        throw 'Error al construir la imagen de MASt3R.'
    }
}

if (-not (Test-Path $VfeDir)) {
    throw "No se encontró el directorio de VideoFrameExtractor en $VfeDir"
}

if ($RebuildColmap) {
    Build-ColmapDocker -Rebuild
    Write-Info 'Reconstrucción de COLMAP completada'
    return
}

Install-Ffmpeg
Install-VideoFrameExtractor
Install-CloudCompare
Build-ColmapDocker
Build-AceDocker
Install-Mast3rWeights
Build-Mast3rDocker

Write-Info 'Instalación completada'