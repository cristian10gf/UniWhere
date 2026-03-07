# Pipeline COLMAP + ACE en UniWhere

Este documento describe las herramientas creadas para integrar la reconstruccion 3D (COLMAP) con la relocalizacion visual (ACE) en un flujo de trabajo unificado.

## Arquitectura general

```text
videos/*.mp4
    |
    v
VideoFrameExtractor (extraccion de frames)
    |
    v
data/<serie>/images/  (frames prefijados por serie)
    |
    v
run-parallel-colmap.sh (COLMAP en paralelo por serie)
    |
    v
data/<serie>/sparse/0/ + database.db  (features, matches, reconstruccion sparse)
    |
    v
colmap2ace.py (conversion de formato)
    |
    v
data/<serie>/ace/{train,test}/{rgb,poses,calibration}/  (dataset ACE)
    |
    v
ACE Docker run.sh (entrenamiento / evaluacion)
    |
    v
data/<serie>/ace/output/scene.pt  (modelo ACE entrenado)
```

## Herramientas disponibles

| Herramienta | Ubicacion | Funcion |
|---|---|---|
| COLMAP paralelo | [preprocesamiento/pipelines/run-parallel-colmap.sh](../preprocesamiento/pipelines/run-parallel-colmap.sh) | Ejecutar COLMAP sobre multiples series en paralelo |
| Conversor COLMAP a ACE | [preprocesamiento/scripts/colmap2ace.py](../preprocesamiento/scripts/colmap2ace.py) | Convertir salida sparse de COLMAP al formato de dataset ACE |
| ACE Docker | [preprocesamiento/models/ace/docker/](../preprocesamiento/models/ace/docker/) | Dockerfile y wrapper para entrenar/evaluar ACE |
| COLMAP Docker | [preprocesamiento/models/colmap/docker/](../preprocesamiento/models/colmap/docker/) | Dockerfile y wrappers existentes para COLMAP |

## Requisitos previos

- Docker instalado con soporte NVIDIA (ver [uso-colmap-docker.md](uso-colmap-docker.md) para configuracion).
- ffmpeg instalado en el host (necesario para VideoFrameExtractor).
- Imagenes Docker de COLMAP y ACE construidas.

Todas las dependencias se pueden instalar de una vez con el script de instalacion:

```bash
./scripts/install_tools.sh
```

A continuacion se detalla cada paso si prefieres hacerlo manualmente.

### Instalar ffmpeg

ffmpeg es necesario para que VideoFrameExtractor pueda decodificar los videos de entrada.

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Verificar
ffmpeg -version
```

### Construir imagen Docker de COLMAP

La imagen se descarga automaticamente desde Docker Hub si no existe localmente. Si prefieres construirla desde el codigo fuente incluido en el proyecto:

```bash
cd preprocesamiento/models/colmap
docker build -f docker/Dockerfile -t colmap:latest .
```

O simplemente descargar la oficial:

```bash
docker pull colmap/colmap:latest
```

Los scripts (`run-series.sh`, `run-parallel-colmap.sh`) detectan automaticamente si existe `colmap:latest` local o descargan `colmap/colmap:latest`.

### Descargar pesos de ACE

```bash
cd preprocesamiento/models/ace
wget https://storage.googleapis.com/niantic-lon-static/research/ace/ace_models.tar.gz
tar -xzf ace_models.tar.gz ace_encoder_pretrained.pt
rm ace_models.tar.gz
```

El archivo `ace_encoder_pretrained.pt` debe quedar en la raiz de `preprocesamiento/models/ace/` para que el Dockerfile lo incluya en la imagen.

### Construir imagen Docker de ACE

```bash
cd preprocesamiento/models/ace
docker build -f docker/Dockerfile -t ace:latest .
```

La imagen se construye automaticamente si no existe al ejecutar `run.sh`, pero conviene hacerlo una vez por adelantado para verificar que funciona.

## 1. Ejecucion paralela de COLMAP

El script [run-parallel-colmap.sh](../preprocesamiento/pipelines/run-parallel-colmap.sh) ejecuta COLMAP sobre multiples series simultaneamente, controlando la concurrencia para no saturar la GPU.

### Estructura esperada

Cada serie debe tener sus imagenes en `preprocesamiento/data/<serie>/images/`:

```text
preprocesamiento/data/
  serie-1/
    images/
      serie-1__frame000001.jpg
      serie-1__frame000002.jpg
      ...
  serie-2/
    images/
      serie-2__frame000001.jpg
      ...
```

Los nombres de imagen deben ser unicos entre series (prefijo de serie) para permitir el merge posterior de bases de datos COLMAP.

### Uso basico

```bash
cd preprocesamiento/pipelines

# Una serie
./run-parallel-colmap.sh serie-1

# Multiples series, una a la vez (default)
./run-parallel-colmap.sh serie-1 serie-2 serie-3

# Dos series en paralelo (requiere VRAM suficiente o modo CPU)
./run-parallel-colmap.sh serie-1 serie-2 serie-3 --max-parallel 2

# Forzar modo CPU
./run-parallel-colmap.sh serie-1 serie-2 --cpu
```

### Opciones

| Opcion | Default | Descripcion |
|---|---|---|
| `--max-parallel N` | 1 | Maximo de ejecuciones simultaneas |
| `--mode MODE` | advanced | Modo de COLMAP: `automatic` o `advanced` |
| `--data-root PATH` | `preprocesamiento/data` | Carpeta base de datos |
| `--with-dense` | (desactivado) | Incluir reconstruccion densa |
| `--cpu` | (desactivado) | Forzar ejecucion sin GPU |
| `-- args...` | | Argumentos extra pasados a COLMAP |

### Salida

Cada serie genera:

```text
data/<serie>/
  database.db          # Base de datos COLMAP (features + matches)
  sparse/0/            # Reconstruccion sparse (cameras.bin, images.bin, points3D.bin)
  colmap.log           # Log de la ejecucion
```

Por defecto, el modo advanced omite la reconstruccion densa (`--no-dense`) para ahorrar tiempo. Usa `--with-dense` si la necesitas.

### Argumentos extra de COLMAP

Cualquier argumento despues de `--` se pasa directamente a `run-series.sh`:

```bash
# Aumentar features por imagen
./run-parallel-colmap.sh serie-1 serie-2 -- --max-features 16384

# Matcher exhaustivo en vez de secuencial
./run-parallel-colmap.sh serie-1 serie-2 -- --matcher exhaustive
```

### Log de ejecucion

Cada serie tiene su log en `data/<serie>/colmap.log`. Al terminar, el script imprime un resumen:

```text
========================================
 Resumen
========================================
Total series : 3
Exitosas     : 2
Fallidas     : 1
Series con error:
  - serie-3 (ver .../data/serie-3/colmap.log)
========================================
```

## 2. Conversion COLMAP a ACE

El script [colmap2ace.py](../preprocesamiento/scripts/colmap2ace.py) convierte la salida sparse de COLMAP al formato de dataset que ACE espera para entrenamiento y evaluacion.

### Que hace la conversion

1. Lee `cameras.txt` e `images.txt` del directorio sparse de COLMAP.
2. Convierte cada pose de world-to-camera (formato COLMAP: quaternion + translation) a camera-to-world (formato ACE: matriz 4x4).
3. Extrae la calibracion segun el modelo de camara COLMAP:
   - Modelos con focal unico (SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL) se escriben como un float.
   - Modelos con fx/fy separados (PINHOLE, OPENCV) se escriben como matriz 3x3.
4. Copia (o crea symlinks a) las imagenes RGB.
5. Divide aleatoriamente en train/test (80/20 por defecto, seed fija).

### Uso

```bash
cd preprocesamiento

# Conversion basica
python scripts/colmap2ace.py \
  --colmap-dir data/serie-1 \
  --output-dir data/serie-1/ace

# Con ratio de entrenamiento diferente
python scripts/colmap2ace.py \
  --colmap-dir data/serie-1 \
  --output-dir data/serie-1/ace \
  --train-ratio 0.9

# Usar symlinks en vez de copiar imagenes (ahorra espacio)
python scripts/colmap2ace.py \
  --colmap-dir data/serie-1 \
  --output-dir data/serie-1/ace \
  --symlink
```

### Opciones

| Opcion | Default | Descripcion |
|---|---|---|
| `--colmap-dir` | (requerido) | Directorio de la serie COLMAP (contiene `sparse/` e `images/`) |
| `--output-dir` | (requerido) | Directorio de salida para el dataset ACE |
| `--train-ratio` | 0.8 | Fraccion de imagenes para entrenamiento |
| `--seed` | 42 | Seed para reproducibilidad del split |
| `--symlink` | false | Crear symlinks a imagenes en vez de copiarlas |

### Salida

```text
data/<serie>/ace/
  train/
    rgb/             # Imagenes de entrenamiento
    poses/           # Matrices 4x4 camera-to-world (.txt)
    calibration/     # Focal length (float) o intrinsics 3x3 (.txt)
  test/
    rgb/
    poses/
    calibration/
```

### Conversion automatica de formato binario

Si COLMAP genero archivos en formato binario (`.bin`), el script intenta convertirlos automaticamente a texto usando `colmap model_converter`. Si COLMAP no esta instalado localmente, lo ejecuta via Docker.

### Modelos de camara soportados

| Modelo COLMAP | Salida ACE |
|---|---|
| SIMPLE_PINHOLE | focal length (float) |
| SIMPLE_RADIAL | focal length (float) |
| RADIAL | focal length (float) |
| SIMPLE_RADIAL_FISHEYE | focal length (float) |
| RADIAL_FISHEYE | focal length (float) |
| PINHOLE | matriz intrinsics 3x3 |
| OPENCV | matriz intrinsics 3x3 |
| OPENCV_FISHEYE | matriz intrinsics 3x3 |
| FULL_OPENCV | matriz intrinsics 3x3 |

## 3. Entrenamiento y evaluacion con ACE Docker

El wrapper [run.sh](../preprocesamiento/models/ace/docker/run.sh) ejecuta ACE dentro de un contenedor Docker con deteccion automatica de GPU.

### Uso

```bash
cd preprocesamiento/models/ace/docker

# Entrenar un modelo ACE sobre una escena
./run.sh train serie-1/ace output/serie-1.pt

# Evaluar el modelo
./run.sh test serie-1/ace output/serie-1.pt
```

Las rutas son relativas a `preprocesamiento/data/`. Es decir, `serie-1/ace` se resuelve como `preprocesamiento/data/serie-1/ace`.

### Opciones

| Opcion | Default | Descripcion |
|---|---|---|
| `--data-root PATH` | `preprocesamiento/data` | Carpeta base de datos |
| `--cpu` | (desactivado) | Forzar ejecucion sin GPU |
| `-- args...` | | Argumentos extra pasados a ACE |

### Argumentos extra de ACE

Cualquier argumento despues de `--` se pasa directamente al script de entrenamiento o evaluacion:

```bash
# Entrenar con mas bloques en el head
./run.sh train serie-1/ace output/serie-1.pt -- --num_head_blocks 2

# Entrenar con mas epocas
./run.sh train serie-1/ace output/serie-1.pt -- --epochs 32

# Evaluar con mas hipotesis RANSAC
./run.sh test serie-1/ace output/serie-1.pt -- --hypotheses 128
```

### Imagen Docker de ACE

La imagen `ace:latest` se construye automaticamente si no existe. Contiene:

- CUDA 11.8 + Python 3.8 + PyTorch 2.0
- Extension C++ dsacstar compilada (RANSAC para estimacion de pose)
- Todas las dependencias de ACE (numpy, scipy, scikit-image, opencv, etc.)
- Codigo fuente de ACE

La imagen usa un Dockerfile multi-stage: el builder compila dsacstar con los headers de desarrollo, y el runtime solo incluye las librerias necesarias para ejecucion.

## Flujo completo: de video a modelo ACE

Ejemplo con dos series de video:

```bash
cd preprocesamiento

# 1. Extraer frames de los videos (con prefijo de serie)
# (ver documentacion de VideoFrameExtractor)

# 2. Ejecutar COLMAP en paralelo sobre ambas series
./pipelines/run-parallel-colmap.sh serie-1 serie-2

# 3. Convertir resultados COLMAP al formato ACE
python scripts/colmap2ace.py --colmap-dir data/serie-1 --output-dir data/serie-1/ace
python scripts/colmap2ace.py --colmap-dir data/serie-2 --output-dir data/serie-2/ace

# 4. Entrenar ACE
./models/ace/docker/run.sh train serie-1/ace output/serie-1.pt

# 5. Evaluar
./models/ace/docker/run.sh test serie-1/ace output/serie-1.pt
```

## Estructura de datos resultante

Despues de ejecutar todo el pipeline, la carpeta de datos queda asi:

```text
preprocesamiento/data/
  serie-1/
    images/                  # Frames originales
    database.db              # BD COLMAP (features + matches)
    sparse/0/                # Reconstruccion sparse
    colmap.log               # Log de COLMAP
    ace/
      train/
        rgb/                 # Imagenes de entrenamiento
        poses/               # Poses camera-to-world (4x4)
        calibration/         # Focal length o intrinsics 3x3
      test/
        rgb/
        poses/
        calibration/
  serie-2/
    images/
    database.db
    sparse/0/
    colmap.log
    ace/
      ...
  output/
    serie-1.pt               # Modelo ACE entrenado
    serie-2.pt
```

## Troubleshooting

### Docker falla por contexto

```bash
docker context use default
```

### COLMAP no detecta GPU

Verificar que el NVIDIA Container Toolkit esta instalado:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime-ubuntu20.04 nvidia-smi
```

Si falla, ejecutar el setup:

```bash
preprocesamiento/models/colmap/docker/setup-ubuntu.sh
```

### colmap2ace.py no encuentra modelo sparse

El script busca en `<serie>/sparse/0/` por defecto. Si COLMAP genero multiples modelos (`sparse/1/`, `sparse/2/`, etc.), el script toma el primero que encuentre. Puedes verificar cual modelo tiene mas imagenes registradas con:

```bash
# Dentro del contenedor COLMAP
colmap model_analyzer --path sparse/0
```

### ACE falla con "encoder not found"

Asegurate de que `ace_encoder_pretrained.pt` esta en `preprocesamiento/models/ace/` antes de construir la imagen Docker. Si ya construiste la imagen sin el encoder, reconstruye:

```bash
cd preprocesamiento/models/ace
docker build -f docker/Dockerfile -t ace:latest --no-cache .
```

### COLMAP paralelo: VRAM insuficiente

Con una sola GPU, usa `--max-parallel 1` (el default). Si tienes multiples GPUs o suficiente VRAM, puedes aumentarlo. Para procesamiento solo en CPU:

```bash
./run-parallel-colmap.sh serie-1 serie-2 --cpu --max-parallel 4
```
