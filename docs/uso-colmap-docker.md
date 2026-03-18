# Flujo de COLMAP con Docker en UniWhere

Este proyecto deja un wrapper para ejecutar COLMAP de forma consistente sobre las series de imagenes guardadas en [preprocesamiento/data](../preprocesamiento/data).

Desde marzo de 2026, el pipeline principal usa MASt3R por defecto (`--reconstructor mast3r`) y COLMAP queda como fallback explicito de compatibilidad (`--reconstructor colmap`).

Este documento cubre el camino COLMAP para fallback, recuperacion y comparativas A/B.

## Convivencia con MASt3R en el pipeline

Ejecutar una corrida completa forzando COLMAP:

```bash
cd preprocesamiento/pipelines
./pipeline.sh --series campus-norte.mp4 --merge --run-ace --reconstructor colmap
```

Serie ya extraida (sin paso de videos):

```bash
./pipeline.sh --series-names block_g_1200 --reconstructor colmap
```

El contrato de salida se mantiene igual que MASt3R:

```text
data/<serie>/database.db
data/<serie>/sparse/0/{cameras,images,points3D}.{txt|bin}
data/<serie>/dense/0/fused.ply
```

## Estructura esperada

Cada serie debe vivir dentro de su propia carpeta y las imagenes deben quedar en una subcarpeta llamada images.

```text
preprocesamiento/
  data/
    edificio-a/
      images/
        frame_0001.jpg
        frame_0002.jpg
        ...
```

Los resultados de COLMAP se escriben en esa misma carpeta de la serie.

```text
preprocesamiento/
  data/
    edificio-a/
      images/
      database.db
      sparse/
      dense/
```

La carpeta [preprocesamiento/data/.gitignore](../preprocesamiento/data/.gitignore) ignora esas imagenes y salidas para no versionarlas en Git.

## Script principal

El wrapper del proyecto esta en [preprocesamiento/models/colmap/docker/run-series.sh](../preprocesamiento/models/colmap/docker/run-series.sh).

Tambien hay un acortador para no tener que navegar hasta la carpeta docker:

- desde la carpeta [preprocesamiento](../preprocesamiento): [preprocesamiento/run-colmap.sh](../preprocesamiento/run-colmap.sh)

Ejemplo minimo:

```bash
cd /home/cristian/Documentos/proyectos/UniWhere/preprocesamiento/models/colmap/docker
./run-series.sh edificio-a
```

Eso ejecuta:

```bash
colmap automatic_reconstructor --image_path ./images --workspace_path .
```

pero montando automaticamente la carpeta [preprocesamiento/data/edificio-a](../preprocesamiento/data) dentro del contenedor.

Tambien puedes ejecutarlo de forma mas corta desde preprocesamiento:

```bash
cd /home/cristian/Documentos/proyectos/UniWhere/preprocesamiento
./run-colmap.sh edificio-a
```

## Modos disponibles

### 1. Reconstruccion automatica

Es el modo por defecto y el mas comodo para una primera corrida.

```bash
./run-series.sh edificio-a
```

Tambien puedes pasar argumentos extra de COLMAP al final:

```bash
./run-series.sh edificio-a -- --quality high
```

### 2. Pipeline avanzado paso a paso

Si necesitas controlar cache, matcher, resolucion o reanudar etapas, usa el modo advanced. Este wrapper delega en [preprocesamiento/models/colmap/docker/run-advance.sh](../preprocesamiento/models/colmap/docker/run-advance.sh).

```bash
./run-series.sh edificio-a --mode advanced -- --cache-size 16 --max-image-size 4000
```

Ejemplos utiles:

```bash
./run-series.sh edificio-a --mode advanced -- --matcher sequential --overlap 20
./run-series.sh edificio-a --mode advanced -- --skip-to dense
./run-series.sh edificio-a --mode advanced -- --no-dense
```

### 3. Shell interactiva dentro del contenedor

Si quieres correr comandos manualmente:

```bash
./run-series.sh edificio-a --mode shell
```

Dentro del contenedor puedes ejecutar:

```bash
colmap automatic_reconstructor --image_path ./images --workspace_path .

colmap feature_extractor --database_path ./database.db --image_path ./images
colmap exhaustive_matcher --database_path ./database.db
colmap mapper --database_path ./database.db --image_path ./images --output_path ./sparse
```

## Error frecuente de Docker Desktop

Si Docker falla por el contexto, ejecuta:

```bash
docker context use default
```

Tambien puedes pedirle al wrapper que lo haga antes de correr:

```bash
./run-series.sh edificio-a --force-context-default
```

## Requisitos

- Docker instalado.
- Si quieres aceleracion densa con GPU, Docker debe tener acceso a NVIDIA.
- La serie debe existir en [preprocesamiento/data](../preprocesamiento/data) con una subcarpeta images.

## Cuándo usar fallback COLMAP

- Si MASt3R falla en una escena particular.
- Si necesitas `vocab_tree_matcher` exacto en merge multi-serie.
- Si quieres comparar calidad de pose MASt3R vs COLMAP con el mismo downstream ACE.

## Visualizacion con CloudCompare

Para abrir los resultados guardados dentro de una serie en [preprocesamiento/data](../preprocesamiento/data), usa el atajo [preprocesamiento/run-cloudcompare.sh](../preprocesamiento/run-cloudcompare.sh).

Ejemplo basico:

```bash
cd /home/cristian/Documentos/proyectos/UniWhere/preprocesamiento
./run-cloudcompare.sh edificio-a
```

Por defecto intenta abrir, en este orden:

- dense/0/fused.ply
- dense/0/meshed-poisson.ply
- dense/0/meshed-delaunay.ply
- el primer archivo .ply, .obj o .stl encontrado en la serie

Opciones utiles:

```bash
./run-cloudcompare.sh edificio-a --mesh
./run-cloudcompare.sh edificio-a --all
./run-cloudcompare.sh edificio-a -- dense/0/fused.ply dense/0/meshed-poisson.ply
```

El lanzador detecta CloudCompare en estos escenarios:

- binario CloudCompare en PATH
- binario ccViewer en PATH
- instalacion Flatpak como org.cloudcompare.CloudCompare

Si no lo tienes instalado, puedes usar el instalador del proyecto:

```bash
./scripts/install_tools.sh
```

## Exportar resultados a ZIP

Si quieres compartir una serie ya procesada por COLMAP sin incluir las imagenes originales, usa [preprocesamiento/scripts/export-colmap-results.sh](../preprocesamiento/scripts/export-colmap-results.sh).

El script:

- pide solo el nombre de la serie
- asume que existe en [preprocesamiento/data](../preprocesamiento/data)
- genera un archivo llamado <serie>-colmap-results.zip dentro de esa misma carpeta
- excluye la carpeta images y archivos de imagen comunes

Uso:

```bash
cd /home/cristian/Documentos/proyectos/UniWhere/preprocesamiento
./scripts/export-colmap-results.sh
```

Luego ingresa, por ejemplo:

```text
edificio-a
```

## Dataset de ejemplo

Si necesitas una fuente publica para pruebas, puedes revisar el dataset visual-inertial de TUM:

https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset
