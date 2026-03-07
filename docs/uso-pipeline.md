# Pipeline COLMAP + ACE en UniWhere

Este documento describe las herramientas creadas para integrar la reconstruccion 3D (COLMAP) con la relocalizacion visual (ACE) en un flujo de trabajo unificado.

## Arquitectura general

```text
videos/*.mp4
    |
    v
VideoFrameExtractor (extraccion de frames con prefijo de serie)
    |
    v
data/<serie>/images/  (frames: serie__frame000001.jpg, ...)
    |
    v
run-parallel-colmap.sh (COLMAP en paralelo por serie)
    |
    v
data/<serie>/sparse/0/ + database.db  (features, matches, reconstruccion sparse)
    |
    v
merge-colmap.sh (database_merger + cross-match + mapper unificado)
    |
    v
data/_merged/sparse/0/ + database.db  (reconstruccion unificada)
    |
    v
colmap2ace.py (conversion de formato)
    |
    v
data/_merged/ace/{train,test}/{rgb,poses,calibration}/  (dataset ACE)
    |
    v
ACE Docker run.sh (entrenamiento / evaluacion)
    |
    v
data/_merged/ace/output/scene.pt  (modelo ACE entrenado)
```

Todo el flujo se puede ejecutar con un solo comando usando `pipeline.sh`, o paso a paso con las herramientas individuales.

## Herramientas disponibles

| Herramienta | Ubicacion | Funcion |
|---|---|---|
| Pipeline completo | [preprocesamiento/pipelines/pipeline.sh](../preprocesamiento/pipelines/pipeline.sh) | Orquestador end-to-end: video -> frames -> COLMAP -> merge -> ACE |
| COLMAP paralelo | [preprocesamiento/pipelines/run-parallel-colmap.sh](../preprocesamiento/pipelines/run-parallel-colmap.sh) | Ejecutar COLMAP sobre multiples series en paralelo |
| Merge COLMAP | [preprocesamiento/pipelines/merge-colmap.sh](../preprocesamiento/pipelines/merge-colmap.sh) | Merge hibrido de reconstrucciones COLMAP |
| Conversor COLMAP a ACE | [preprocesamiento/scripts/colmap2ace.py](../preprocesamiento/scripts/colmap2ace.py) | Convertir salida sparse de COLMAP al formato de dataset ACE |
| ACE Docker | [preprocesamiento/models/ace/docker/](../preprocesamiento/models/ace/docker/) | Dockerfile y wrapper para entrenar/evaluar ACE |
| ACE shortcut | [preprocesamiento/run-ace.sh](../preprocesamiento/run-ace.sh) | Wrapper de conveniencia para ACE (como `run-colmap.sh`) |
| ACE Viewer | [preprocesamiento/run-ace-viewer.sh](../preprocesamiento/run-ace-viewer.sh) | Visualizador interactivo de resultados ACE con Rerun |
| COLMAP Docker | [preprocesamiento/models/colmap/docker/](../preprocesamiento/models/colmap/docker/) | Dockerfile y wrappers existentes para COLMAP |

## Requisitos previos

- Docker instalado con soporte NVIDIA (ver [uso-colmap-docker.md](uso-colmap-docker.md) para configuracion).
- [uv](https://docs.astral.sh/uv/) instalado (gestor de paquetes Python).
- `videoframeextractor` instalado como tool de uv (ver mas abajo).
- ffmpeg instalado en el host (necesario para VideoFrameExtractor y para detectar dimensiones de video).
- Imagenes Docker de COLMAP y ACE construidas.
- Python 3 con `numpy` y `scipy` (para `colmap2ace.py`).

Todas las dependencias se pueden instalar de una vez con el script de instalacion:

```bash
./scripts/install_tools.sh
```

A continuacion se detalla cada paso si prefieres hacerlo manualmente.

### Instalar uv

uv es necesario para instalar VideoFrameExtractor como herramienta CLI global.

```bash
# Instalacion recomendada
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verificar
uv --version
```

### Instalar VideoFrameExtractor

VideoFrameExtractor se instala como tool de uv, lo que lo hace disponible como comando global `videoframeextractor`:

```bash
uv tool install preprocesamiento/VideoFrameExtractor

# Verificar
videoframeextractor --help
```

Si el comando no aparece despues de instalar, abre una nueva terminal o ejecuta `uv tool update-shell`.

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

Los scripts (`run-series.sh`, `run-parallel-colmap.sh`, `merge-colmap.sh`) detectan automaticamente si existe `colmap:latest` local o descargan `colmap/colmap:latest`.

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

### Descargar vocabulary tree (para merge multi-serie)

Si vas a hacer merge de multiples series con `vocab_tree_matcher` (recomendado):

```bash
cd preprocesamiento/data
mkdir -p _merged
# Elegir segun cantidad de imagenes:
#   < 1000 imagenes : vocab_tree_flickr100K_words32K.bin
#   1K-10K imagenes : vocab_tree_flickr100K_words256K.bin
#   > 10K imagenes  : vocab_tree_flickr100K_words1M.bin
wget -O _merged/vocab_tree.bin \
  https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin
```

Alternativa: usar `--matcher exhaustive` para evitar la necesidad del vocabulary tree (mas lento pero sin archivos extra).

## Pipeline completo con un solo comando

El script [pipeline.sh](../preprocesamiento/pipelines/pipeline.sh) orquesta todo el flujo de principio a fin.

### De video a modelo ACE

Los videos se buscan por defecto en `preprocesamiento/videos/`.

```bash
cd preprocesamiento/pipelines

# Pipeline completo: video -> frames -> COLMAP -> merge -> ACE
./pipeline.sh \
  --series campus-norte.mp4 campus-sur.mp4 campus-este.mp4 \
  --merge \
  --run-ace

# Videos en otra carpeta
./pipeline.sh \
  --videos-dir /ruta/a/videos \
  --series campus-norte.mp4 \
  --merge --run-ace

# Solo video -> frames -> COLMAP (sin merge ni ACE)
./pipeline.sh --series video1.mp4
```

El pipeline detecta automaticamente las dimensiones de cada video con `ffprobe` y las pasa a `videoframeextractor` con `-w` y `-H`, de modo que los frames mantienen la resolucion original sin redimensionar ni agregar padding. Esto es importante para que COLMAP trabaje con la geometria real de la camara. Se usa GPU por defecto para la decodificacion (`--gpu`).

### Solo merge de series ya procesadas

```bash
./pipeline.sh --merge-only serie-1 serie-2 serie-3
```

### Solo ACE sobre datos ya convertidos

```bash
./pipeline.sh --ace-only _merged/ace
```

### Opciones del pipeline

| Opcion | Default | Descripcion |
|---|---|---|
| `--videos-dir PATH` | `preprocesamiento/videos` | Carpeta con videos de entrada |
| `--series FILE [...]` | | Videos a procesar (relativos a --videos-dir) |
| `--series-names NAME [...]` | | Series ya extraidas (sin extraccion de frames) |
| `--sample-fps FPS` | 2 | FPS de muestreo para extraccion |
| `--max-parallel N` | 1 | Maximo de COLMAPs en paralelo |
| `--colmap-mode MODE` | advanced | Modo COLMAP: `automatic` o `advanced` |
| `--merge` | (desactivado) | Ejecutar merge tras COLMAP individual |
| `--merge-only S [...]` | | Solo merge (sin extraccion ni COLMAP) |
| `--matcher TYPE` | vocab_tree | Matcher para cross-serie: `vocab_tree` o `exhaustive` |
| `--vocab-tree PATH` | | Ruta al vocabulary tree |
| `--run-ace` | (desactivado) | Ejecutar conversion + entrenamiento ACE |
| `--ace-only PATH` | | Solo ACE sobre datos ya convertidos |
| `--train-ratio RATIO` | 0.8 | Ratio train/test para colmap2ace |
| `--cpu` | (desactivado) | Forzar modo CPU en todos los pasos |

La extraccion de frames usa `videoframeextractor` (instalado como tool de uv) con GPU y las dimensiones nativas de cada video (detectadas via `ffprobe`).

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

## 2. Merge de reconstrucciones COLMAP

El script [merge-colmap.sh](../preprocesamiento/pipelines/merge-colmap.sh) crea una reconstruccion unificada a partir de multiples series, reutilizando todo el trabajo previo de feature extraction y matching.

### Como funciona (enfoque hibrido)

El enfoque reutiliza la computacion GPU-heavy (features, matches) de los COLMAPs individuales:

1. **Merge de bases de datos** (`database_merger`): fusiona secuencialmente las N bases de datos, preservando keypoints, descriptors, matches y two-view geometries intra-serie.
2. **Symlinks de imagenes**: crea `_merged/images/` con symlinks a las imagenes de todas las series.
3. **Matching cross-serie**: ejecuta el matcher sobre la BD unificada. COLMAP salta automaticamente los pares intra-serie que ya tienen matches.
4. **Mapper unificado**: reconstruccion SfM completa sobre la BD con todos los matches.

| Operacion | Desde cero | Enfoque hibrido |
|---|---|---|
| Feature extraction (GPU) | Serial, todas las imagenes | Paralelo por serie (ya hecho) |
| Matching intra-serie (GPU) | Serial, todos los pares | Paralelo por serie (ya hecho) |
| Matching cross-serie | N/A | Solo pares nuevos |
| Mapper (CPU) | 1 ejecucion | 1 ejecucion |

### Uso

```bash
cd preprocesamiento/pipelines

# Merge basico (vocab_tree_matcher)
./merge-colmap.sh serie-1 serie-2 serie-3

# Con matcher exhaustivo (sin necesidad de vocabulary tree)
./merge-colmap.sh serie-1 serie-2 --matcher exhaustive

# Especificar vocabulary tree
./merge-colmap.sh serie-1 serie-2 --vocab-tree _merged/vocab_tree.bin

# Con reconstruccion densa
./merge-colmap.sh serie-1 serie-2 --with-dense

# Reanudar desde el matching (si el merge de BDs ya se hizo)
./merge-colmap.sh serie-1 serie-2 --skip-to matching
```

### Opciones

| Opcion | Default | Descripcion |
|---|---|---|
| `--data-root PATH` | `preprocesamiento/data` | Carpeta base de datos |
| `--output-name NAME` | `_merged` | Nombre de la carpeta de salida |
| `--matcher TYPE` | `vocab_tree` | `vocab_tree` o `exhaustive` |
| `--vocab-tree PATH` | | Ruta al vocabulary tree (relativa a data-root) |
| `--with-dense` | (desactivado) | Incluir reconstruccion densa |
| `--cpu` | (desactivado) | Forzar modo CPU |
| `--skip-to STAGE` | | Reanudar desde: `merge`, `symlinks`, `matching`, `mapper`, `dense` |

### Salida

```text
data/_merged/
  database.db              # BD unificada (features + matches intra + cross)
  images/                  # Symlinks a imagenes de todas las series
  sparse/0/                # Reconstruccion sparse unificada
  dense/0/                 # (si --with-dense) Reconstruccion densa
    fused.ply
```

### Requisitos para el merge

- Los COLMAPs individuales deben haberse ejecutado previamente (cada serie con `database.db`).
- Las imagenes deben tener nombres unicos entre series (prefijo de serie, ej: `serie-1__frame000001.jpg`).
- Debe existir solapamiento fisico real entre las capturas de distintas series para que el matching cross-serie encuentre correspondencias.
- Para `vocab_tree_matcher`: descargar un vocabulary tree de https://demuc.de/colmap/ (ver seccion de requisitos previos).

## 3. Conversion COLMAP a ACE

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

# Conversion basica (serie individual)
python scripts/colmap2ace.py \
  --colmap-dir data/serie-1 \
  --output-dir data/serie-1/ace

# Conversion del modelo merged
python scripts/colmap2ace.py \
  --colmap-dir data/_merged \
  --output-dir data/_merged/ace

# Con ratio de entrenamiento diferente
python scripts/colmap2ace.py \
  --colmap-dir data/_merged \
  --output-dir data/_merged/ace \
  --train-ratio 0.9

# Usar symlinks en vez de copiar imagenes (ahorra espacio)
python scripts/colmap2ace.py \
  --colmap-dir data/_merged \
  --output-dir data/_merged/ace \
  --symlink
```

### Opciones

| Opcion | Default | Descripcion |
|---|---|---|
| `--colmap-dir` | (requerido) | Directorio COLMAP (contiene `sparse/` e `images/`) |
| `--output-dir` | (requerido) | Directorio de salida para el dataset ACE |
| `--train-ratio` | 0.8 | Fraccion de imagenes para entrenamiento |
| `--seed` | 42 | Seed para reproducibilidad del split |
| `--symlink` | false | Crear symlinks a imagenes en vez de copiarlas |

### Salida

```text
data/<target>/ace/
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

## 4. Entrenamiento y evaluacion con ACE

### Wrapper de conveniencia

El archivo [run-ace.sh](../preprocesamiento/run-ace.sh) es un shortcut que delega a `models/ace/docker/run.sh`, analogo a `run-colmap.sh`:

```bash
cd preprocesamiento

# Entrenar
./run-ace.sh train serie-1/ace output/serie-1.pt

# Evaluar
./run-ace.sh test serie-1/ace output/serie-1.pt

# Sobre el modelo merged
./run-ace.sh train _merged/ace output/_merged.pt
./run-ace.sh test _merged/ace output/_merged.pt
```

### Uso directo del wrapper Docker

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
./run-ace.sh train serie-1/ace output/serie-1.pt -- --num_head_blocks 2

# Entrenar con mas epocas
./run-ace.sh train serie-1/ace output/serie-1.pt -- --epochs 32

# Evaluar con mas hipotesis RANSAC
./run-ace.sh test serie-1/ace output/serie-1.pt -- --hypotheses 128
```

### Imagen Docker de ACE

La imagen `ace:latest` se construye automaticamente si no existe. Contiene:

- CUDA 11.8 + Python 3.8 + PyTorch 2.0
- Extension C++ dsacstar compilada (RANSAC para estimacion de pose)
- Todas las dependencias de ACE (numpy, scipy, scikit-image, opencv, etc.)
- Codigo fuente de ACE

La imagen usa un Dockerfile multi-stage: el builder compila dsacstar con los headers de desarrollo, y el runtime solo incluye las librerias necesarias para ejecucion.

## 5. Visualizacion de resultados ACE

El visualizador interactivo [ace-rerun](../preprocesamiento/visualizadores/ace-rerun/) permite inspeccionar los resultados de ACE en 3D usando [Rerun](https://rerun.io/). Carga la nube de puntos original de COLMAP (la reconstruccion densa/sparse con la que se entreno ACE) y la muestra junto con las poses de camara estimadas por ACE.

### Que muestra

- **Nube de puntos COLMAP** coloreada (densa o sparse)
- **Frustums de camaras de mapping** (entrenamiento)
- **Poses estimadas por ACE** coloreadas por error (verde=bueno, rojo=malo)
- **Trayectoria ground truth** de test en azul
- **Imagenes de query** en el frustum de la pose estimada
- **Series temporales** de error de rotacion, traslacion e inliers
- **Timeline interactiva** para navegar por los frames de test

### Wrapper de conveniencia

El archivo [run-ace-viewer.sh](../preprocesamiento/run-ace-viewer.sh) es un shortcut analogo a `run-cloudcompare.sh`:

```bash
cd preprocesamiento

# Visualizar con auto-deteccion de nube COLMAP
./run-ace-viewer.sh _merged

# Con resultados de test ACE (poses estimadas)
./run-ace-viewer.sh _merged --test-poses data/output/poses__merged.txt

# Especificar PLY manualmente
./run-ace-viewer.sh serie-1 --point-cloud data/serie-1/dense/0/fused.ply

# Exportar nube a PLY para abrir en CloudCompare
./run-ace-viewer.sh _merged --export-ply nube_colmap.ply
```

### Uso directo con uv

```bash
cd preprocesamiento/visualizadores/ace-rerun

# Auto-detectar nube desde directorio COLMAP
uv run visualize_ace.py \
  --colmap-dir ../../data/_merged \
  --scene ../../data/_merged/ace \
  --test-poses ../../data/output/poses__merged.txt

# Con PLY directo
uv run visualize_ace.py \
  --point-cloud ../../data/_merged/dense/0/fused.ply \
  --scene ../../data/_merged/ace

# Agregar nube adicional extraida de la red ACE (para comparar)
uv run visualize_ace.py \
  --colmap-dir ../../data/_merged \
  --scene ../../data/_merged/ace \
  --model ../../data/output/_merged.pt \
  --test-poses ../../data/output/poses__merged.txt
```

### Opciones

| Opcion | Default | Descripcion |
|---|---|---|
| `--point-cloud FILE` | | Path directo a PLY (e.g. `dense/0/fused.ply`) |
| `--colmap-dir PATH` | | Directorio COLMAP (auto-detecta nube de puntos) |
| `--scene PATH` | (requerido) | Directorio dataset ACE (con `train/` y `test/`) |
| `--test-poses FILE` | | Archivo de resultados ACE (`poses_*.txt`) |
| `--model FILE` | | Head entrenado (`.pt`) para nube adicional de la red |
| `--encoder FILE` | (auto-detectado) | Encoder pre-entrenado |
| `--max-points N` | 1000000 | Maximo de puntos a visualizar |
| `--export-ply FILE` | | Exportar nube a PLY para CloudCompare |

### Deteccion automatica de nube de puntos

Con `--colmap-dir`, el visualizador busca en este orden:
1. `dense/0/fused.ply` (reconstruccion densa)
2. `dense/fused.ply`
3. `sparse/0/points3D.txt` (reconstruccion sparse)
4. `sparse/0/points3D.bin`

### Estructura del paquete

```text
preprocesamiento/visualizadores/ace-rerun/
  pyproject.toml           # Dependencias (uv)
  visualize_ace.py         # Entry point (CLI + orquestacion)
  ace_rerun/
    __init__.py
    point_cloud.py         # Carga de nubes COLMAP (PLY, points3D.txt/bin)
    ace_extraction.py      # Extraccion de nube desde red ACE
    poses.py               # Carga de poses, parsing de resultados ACE
    viewer.py              # Visualizacion Rerun y exportacion PLY
```

### Requisitos

- Python >= 3.10 y [uv](https://docs.astral.sh/uv/) instalado.
- Las dependencias se instalan automaticamente con `uv sync` o al primer `uv run`.
- GPU con CUDA solo es necesaria si se usa `--model` para extraer nube de la red ACE.

## Flujo completo paso a paso

Ejemplo con dos series de video, ejecutado manualmente paso a paso:

```bash
cd preprocesamiento

# 1. Detectar dimensiones y extraer frames (resolucion original, con GPU)
# Primero obtener dimensiones del video:
ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
  -of csv=s=x:p=0 videos/campus-norte.mp4
# Ejemplo de salida: 1920x1080

# Extraer frames pasando las dimensiones originales:
videoframeextractor videos/campus-norte.mp4 \
  -o data/campus-norte/images/ --sample-fps 2 -w 1920 -H 1080 --gpu -f jpg

videoframeextractor videos/campus-sur.mp4 \
  -o data/campus-sur/images/ --sample-fps 2 -w 1920 -H 1080 --gpu -f jpg

# Prefijar manualmente para garantizar nombres unicos entre series:
for f in data/campus-norte/images/*; do
  mv "$f" "data/campus-norte/images/campus-norte__$(basename "$f")"
done
for f in data/campus-sur/images/*; do
  mv "$f" "data/campus-sur/images/campus-sur__$(basename "$f")"
done

# 2. Ejecutar COLMAP en paralelo sobre ambas series
./pipelines/run-parallel-colmap.sh campus-norte campus-sur

# 3. Merge de las reconstrucciones (enfoque hibrido)
./pipelines/merge-colmap.sh campus-norte campus-sur --matcher exhaustive

# 4. Convertir resultados COLMAP al formato ACE
python scripts/colmap2ace.py \
  --colmap-dir data/_merged \
  --output-dir data/_merged/ace \
  --symlink

# 5. Entrenar ACE
./run-ace.sh train _merged/ace output/_merged.pt

# 6. Evaluar
./run-ace.sh test _merged/ace output/_merged.pt

# 7. Visualizar resultados en 3D
./run-ace-viewer.sh _merged --test-poses data/output/poses__merged.txt
```

O todo de una vez con `pipeline.sh` (los videos se buscan en `preprocesamiento/videos/` por defecto):

```bash
cd preprocesamiento/pipelines

./pipeline.sh \
  --series campus-norte.mp4 campus-sur.mp4 \
  --merge \
  --matcher exhaustive \
  --run-ace
```

## Estructura de datos resultante

Despues de ejecutar todo el pipeline con merge, la carpeta de datos queda asi:

```text
preprocesamiento/data/
  campus-norte/
    images/                  # Frames: campus-norte__frame000001.jpg, ...
    database.db              # BD COLMAP individual (features + matches intra-serie)
    sparse/0/                # Reconstruccion sparse individual
    colmap.log               # Log de COLMAP
  campus-sur/
    images/                  # Frames: campus-sur__frame000001.jpg, ...
    database.db
    sparse/0/
    colmap.log
  _merged/
    images/                  # Symlinks a imagenes de TODAS las series
    database.db              # BD unificada (features + matches intra + cross)
    vocab_tree.bin           # Vocabulary tree (si se usa vocab_tree_matcher)
    sparse/0/                # Reconstruccion sparse unificada
    dense/0/                 # (opcional) Reconstruccion densa
      fused.ply
    ace/
      train/
        rgb/                 # Imagenes de entrenamiento
        poses/               # Poses camera-to-world (4x4)
        calibration/         # Focal length o intrinsics 3x3
      test/
        rgb/
        poses/
        calibration/
  output/
    _merged.pt               # Modelo ACE entrenado
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

### Merge: vocab_tree_matcher requiere archivo de vocabulary tree

Si usas `--matcher vocab_tree` (default), necesitas descargar el vocabulary tree:

```bash
cd preprocesamiento/data/_merged
wget https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin -O vocab_tree.bin
```

Alternativa: usar `--matcher exhaustive` (sin archivo extra, pero mas lento para datasets grandes).

### Merge: no se encuentran correspondencias cross-serie

Para que el matching cross-serie funcione, debe existir solapamiento fisico real entre las capturas de distintas series. Si las series cubren areas completamente distintas, el mapper las reconstruira como componentes separados dentro del modelo unificado.

### Pipeline: frames no se prefijan correctamente

El `pipeline.sh` prefija automaticamente los frames con el nombre de serie (`serie__frame000001.jpg`). Si extraes frames manualmente, asegurate de que los nombres sean unicos globalmente antes de hacer merge.
