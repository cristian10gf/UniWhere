$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$VfeDir = Join-Path $RepoRoot 'preprocesamiento/VideoFrameExtractor'
$ToolsDir = Join-Path $RepoRoot '.tools'
$VenvDir = Join-Path $ToolsDir 'videoframeextractor'
$VfeToolName = 'videoframeextractor'

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

if (-not (Test-Path $VfeDir)) {
    throw "No se encontró el directorio de VideoFrameExtractor en $VfeDir"
}

Install-VideoFrameExtractor
Install-CloudCompare

Write-Info 'Instalación completada'