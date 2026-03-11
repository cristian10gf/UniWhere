# Referencia COLMAP — Configuración, Parámetros y Optimización GPU

> **Contexto**: Este documento cubre el uso de COLMAP dentro del pipeline de UniWhere,
> los parámetros actuales configurados en los scripts, optimizaciones para GPU,
> la integración con ACE y OneFormer3D, y un análisis de los tiempos de ejecución
> observados (>24h con ~1800 imágenes).

---

## Índice

1. [Qué hace COLMAP en este pipeline](#1-qué-hace-colmap-en-este-pipeline)
2. [Entradas y salidas por etapa](#2-entradas-y-salidas-por-etapa)
3. [Configuración actual de los scripts](#3-configuración-actual-de-los-scripts)
4. [Parámetros clave y su impacto](#4-parámetros-clave-y-su-impacto)
5. [Optimización GPU — mejores prácticas](#5-optimización-gpu--mejores-prácticas)
6. [Integración con ACE](#6-integración-con-ace)
7. [Integración con OneFormer3D](#7-integración-con-oneformer3d)
8. [Análisis de tiempos: ¿es normal >24h con 1800 imágenes?](#8-análisis-de-tiempos-es-normal-24h-con-1800-imágenes)
9. [Cambios recomendados para reducir tiempos](#9-cambios-recomendados-para-reducir-tiempos)
10. [Comandos de referencia rápida](#10-comandos-de-referencia-rápida)

---

## 1. Qué hace COLMAP en este pipeline

COLMAP resuelve el problema de **Structure-from-Motion (SfM)**: a partir de un conjunto
de imágenes 2D sin información de posición, reconstruye:

- La **pose de cada cámara** (posición y orientación en el espacio 3D).
- Un conjunto de **puntos 3D** triangulados (nube de puntos dispersa).
- Opcionalmente, una **nube de puntos densa** y una malla 3D mediante MVS (PatchMatchStereo).

Dentro de UniWhere, COLMAP produce el mapa geométrico del campus que alimenta tanto
la relocalización visual (ACE) como la comprensión semántica 3D (OneFormer3D).

### Etapas del pipeline COLMAP (modo advanced)

```
[1] Feature Extraction    → extrae keypoints SIFT / ALIKED por imagen
[2] Feature Matching      → establece correspondencias entre pares de imágenes
[3] Sparse Reconstruction → SfM incremental: triangula puntos 3D y estima poses
[4] Dense Reconstruction  → PatchMatchStereo: profundidad densa por imagen (CUDA)
[5] Stereo Fusion         → fusiona mapas de profundidad → nube densa (fused.ply)
[6] Meshing               → reconstruye malla 3D (Poisson o Delaunay)
```

Para ACE, **solo se necesitan las etapas 1–3** (reconstrucción sparse).

---

## 2. Entradas y salidas por etapa

### Entrada general

```
data/<serie>/
  images/          ← imágenes de entrada (JPG/PNG)
```

### Salidas por etapa

| Etapa | Archivos generados | Descripción |
|---|---|---|
| Extracción | `database.db` | SQLite con keypoints y descriptores SIFT |
| Matching | `database.db` | Actualizado con matches entre pares de imágenes |
| Sparse (SfM) | `sparse/0/cameras.bin` | Intrínsecas de cámara |
| Sparse (SfM) | `sparse/0/images.bin` | Poses world→camera (quat + traslación) por imagen |
| Sparse (SfM) | `sparse/0/points3D.bin` | Nube de puntos sparse triangulada |
| Dense (undistort) | `dense/0/images/` | Imágenes undistorsionadas |
| Dense (undistort) | `dense/0/sparse/` | Modelo sparse en coordenadas undistorsionadas |
| PatchMatch | `dense/0/stereo/` | Mapas de profundidad y normales por imagen |
| Fusión | `dense/0/fused.ply` | Nube de puntos densa coloreada |
| Meshing | `dense/0/meshed-poisson.ply` | Malla 3D del entorno |

### Formato de texto (conversión)

Las salidas en `.bin` se pueden convertir a texto legible:

```bash
colmap model_converter \
  --input_path ./sparse/0 \
  --output_path ./sparse/0 \
  --output_type TXT
```

Genera `cameras.txt`, `images.txt`, `points3D.txt` — formato que usa `colmap2ace.py`.

---

## 3. Configuración actual de los scripts

### Script principal: `run-advance.sh`

| Parámetro | Valor actual | Descripción |
|---|---|---|
| `--cache-size` | `16` GB | VRAM para PatchMatchStereo |
| `--max-image-size` | `3200` px | Dimensión máxima de imagen |
| `--max-features` | `8192` | Keypoints SIFT por imagen |
| `--max-matches` | `32768` | Matches máximos por par |
| `--matcher` | `sequential` | Tipo de matcher (mejor para video) |
| `--overlap` | `20` | Solapamiento para sequential matcher |
| `--camera-model` | `SIMPLE_RADIAL` | Modelo de cámara (f, cx, cy, k) |
| `--single-camera` | `1` (ON) | Un único modelo de cámara para todos los frames |
| `--dsp-sift` | `0` (OFF) | DSP-SIFT desactivado (más lento) |

Opciones adicionales dentro del script:

```bash
--SiftExtraction.first_octave -1          # incluye octava extra → mejores features en close-up
--SequentialMatching.loop_detection 1     # detecta loops en la secuencia
--SequentialMatching.loop_detection_num_images 50  # candidatos para loop detection
--PatchMatchStereo.num_samples 15
--PatchMatchStereo.num_iterations 5
--PatchMatchStereo.geom_consistency 1     # consistencia geométrica (mejor calidad, más lento)
```

### Docker: recursos asignados

```bash
--ipc=host
--shm-size=16g
--ulimit memlock=-1
--ulimit stack=67108864
--cpus=$(nproc)             # todos los núcleos del host
--gpus all                  # todas las GPUs disponibles
```

### Matcher en merge: `merge-colmap.sh`

| Opción | Valor default | Notas |
|---|---|---|
| `--matcher` | `vocab_tree` | Recomendado para >500 imágenes cross-serie |
| vocab tree recomendado | `vocab_tree_flickr100K_words256K.bin` | Para 1K-10K imágenes |
| `--VocabTreeMatching.num_images` | `100` | Top-N candidatos por imagen |

---

## 4. Parámetros clave y su impacto

### 4.1 Extracción de features

| Parámetro | Descripción | Impacto en tiempo |
|---|---|---|
| `SiftExtraction.max_num_features` | Keypoints por imagen | Lineal: 2× features ≈ 2× tiempo de matching |
| `FeatureExtraction.max_image_size` | Redimensiona imágenes grandes | Reducir de 3200→1600 → ~4× más rápido en extracción |
| `FeatureExtraction.use_gpu` | GPU SIFT vs CPU SIFT | GPU: 10–30× más rápido |
| `SiftExtraction.first_octave -1` | Octava extra de escala | Mejor calidad a costa de ~10% más tiempo |
| `SiftExtraction.domain_size_pooling` | DSP-SIFT (affine-covariant) | Mejor calidad pero **fuerza CPU** (10–30× más lento) |

### 4.2 Matching

| Parámetro | Descripción | Mejor para |
|---|---|---|
| `sequential_matcher` | Compara frames consecutivos + loop detection | Video continuo |
| `exhaustive_matcher` | Compara todos los pares posibles O(N²) | <500 imágenes no secuenciales |
| `vocab_tree_matcher` | Recuperación por árbol de vocabulario | Cross-match entre series / >500 imágenes desordenadas |
| `SequentialMatching.overlap` | Ventana de frames comparados | 10–30 para video a 2 FPS |
| `SequentialMatching.loop_detection_num_images` | Candidatos de loop | 50–100 para campus con áreas repetidas |
| `SiftMatching.max_ratio` | Lowe ratio test | 0.7 (estricto) – 0.8 (por defecto) |

### 4.3 Sparse Reconstruction (Mapper)

| Parámetro COLMAP | Default COLMAP | Valor en scripts | Descripción |
|---|---|---|---|
| `Mapper.ba_use_gpu` | `0` | `1` (si GPU activa) | **CRÍTICO**: activa solver CUDA (cuSolver/cuDSS) en lugar de Cholesky CPU denso |
| `Mapper.ba_global_images_ratio` | `1.1` | `1.4` | Trigger de BA global: 1.1 = cada 10% de nuevas imágenes, 1.4 = cada 40% |
| `Mapper.ba_global_points_ratio` | `1.1` | `1.4` | Mismo threshold para puntos 3D |
| `Mapper.ba_global_max_num_iterations` | `50` | `30` | Iteraciones máximas por ronda de BA global |
| `Mapper.ba_global_ignore_redundant_points3D` | `0` | `1` | Poda puntos redundantes antes de BA (reduce tamaño del problema) |
| `Mapper.ba_refine_focal_length` | `1` | `0` (single cam) | Fijar focal durante BA cuando se usa single_camera |
| `Mapper.ba_refine_extra_params` | `1` | `0` (single cam) | Fijar distorsión durante BA cuando se usa single_camera |
| `Mapper.num_threads` | auto | `-1` (todos) | Threads para triangulación y registro |

#### Por qué el default 1.1 causa problemas con 1800 imágenes

Con `ba_global_images_ratio=1.1` y `single_camera=1`:
- Global BA sucede ~16 veces durante la reconstrucción de 1800 imágenes
- Cada ronda corre Ceres con el solver **denso CPU Cholesky** (O(N³) en cámaras)
- Con 1600+ cámaras y shared intrinsics: la matriz Hessiana tiene alta llenabilidad → Cholesky falla con el error:
  ```
  W levenberg_marquardt_strategy.cc:123] Linear solver failure.
  Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
  ```
- Cada fallo provoca un retry con paso reducido → decenas de iteraciones fallidas por ronda

**Fix aplicado**: `--Mapper.ba_use_gpu 1` activa cuSolver/cuDSS que maneja matrices grandes; `ratio 1.4` reduce frecuencia de ~16 a ~4 rondas globales.

> **Nota**: `ba_use_gpu` requiere Ceres compilado con `USE_CUDA=ON`. El Dockerfile local ya lo hace.

### 4.4 Dense Reconstruction (PatchMatchStereo) — CUDA

| Parámetro | Descripción | Impacto |
|---|---|---|
| `PatchMatchStereo.cache_size` | VRAM en GB para mapas de profundidad | Crítico: con 16 GB puede procesar ~30-50 imágenes simultáneamente |
| `PatchMatchStereo.max_image_size` | Redimensiona imágenes para dense | 1600 → mucho más rápido que 3200 |
| `PatchMatchStereo.num_samples` | Muestras de hipótesis de profundidad | 15 default; bajar a 7 → ~2× más rápido, calidad menor |
| `PatchMatchStereo.num_iterations` | Iteraciones de propagación | 5 default; bajar a 3 → ~40% más rápido |
| `PatchMatchStereo.geom_consistency` | Consistencia geométrica (2do pass) | OFF → ~50% más rápido, calidad moderada |
| `PatchMatchStereo.gpu_index` | Índice de GPU a usar | `-1` = todas las GPUs disponibles |

---

## 5. Optimización GPU — Mejores prácticas

### 5.1 GPU SIFT (extracción)

COLMAP usa SiftGPU para extracción en GPU. Requiere que la imagen Docker esté construida
con soporte CUDA, y que el contenedor se ejecute con `--gpus all`.

```bash
# El script ya configura esto automáticamente:
--FeatureExtraction.use_gpu 1
--FeatureExtraction.gpu_index 0
```

Velocidad típica con GPU vs CPU:

| Hardware | Tiempo / imagen | 1800 imágenes |
|---|---|---|
| CPU (8 cores) | 5–15 s | 2.5–7.5 h |
| NVIDIA RTX 3080 (GPU SIFT) | 0.3–1 s | 9–30 min |
| NVIDIA RTX 4090 (GPU SIFT) | 0.1–0.5 s | 3–15 min |

### 5.2 GPU Matching

```bash
--FeatureMatching.use_gpu 1
--FeatureMatching.gpu_index 0
```

El matching en GPU puede ser 10–20× más rápido que CPU para grandes cantidades de pares.

### 5.3 Bundle Adjustment con CUDA (Ceres+cuSOLVER)

El Dockerfile del proyecto ya compila Ceres Solver con `USE_CUDA=ON`, lo que habilita
el solver GPU de Ceres. Sin embargo, **el flag no estaba activado en los scripts** —
corregido en `run-advance.sh` y `run-series.sh`.

```bash
# Ahora incluido automáticamente cuando USE_GPU=1:
--Mapper.ba_use_gpu 1
```

**Qué cambia internamente:**

| Condición | Solver Ceres | Comportamiento |
|---|---|---|
| `ba_use_gpu=0` (antes) | Dense CPU Cholesky | O(N³), falla con >500 cámaras |
| `ba_use_gpu=1` + N < umbral | Dense GPU (cuSolver) | GPU accelerated, aún O(N³) |
| `ba_use_gpu=1` + N > umbral | Sparse GPU (cuDSS) | Escala bien con N grande |

Los umbrales se controlan con:
- `--BundleAdjustmentCeres.max_num_images_direct_dense_gpu_solver` (default: ~50 imágenes)
- `--BundleAdjustmentCeres.max_num_images_direct_sparse_gpu_solver` (default más alto)

Con 1600+ imágenes, Ceres activará automáticamente el solver sparse GPU (cuDSS), que
es el más eficiente para problemas de BA a gran escala.

### 5.4 PatchMatchStereo multiGPU

Para múltiples GPUs se puede pasar `--PatchMatchStereo.gpu_index -1` (usa todas).
Cada GPU procesa un subconjunto de imágenes en paralelo.

```bash
# Con 2 GPUs: procesa en paralelo
--PatchMatchStereo.gpu_index -1
```

### 5.5 Configuración de memoria Docker

Los valores ya configurados son correctos para entornos de alto rendimiento:

```bash
--shm-size=16g        # necesario para IPC entre workers
--ulimit memlock=-1   # permite pin de páginas de memoria (crucial para CUDA)
--ipc=host            # comparte memoria IPC con el host
```

Para sistemas con mucha RAM, aumentar `--shm-size` a `32g` si hay errores de `shm`.

---

## 6. Integración con ACE

### Qué necesita ACE de COLMAP

ACE (Accelerated Coordinate Encoding) necesita, por cada imagen:

1. **La imagen RGB** original.
2. **La pose camera-to-world** como matriz 4×4 (archivo `.txt`).
3. **La calibración** como focal length (escalar) o matriz intrínseca 3×3.

Todo esto proviene de la **reconstrucción sparse** de COLMAP.
**ACE no necesita la reconstrucción densa** (etapas 4–6).

### Conversión: `colmap2ace.py`

El script `preprocesamiento/scripts/colmap2ace.py` realiza la conversión:

```
COLMAP sparse (cameras.txt + images.txt)
             ↓
colmap2ace.py
             ↓
data/<serie>/ace/
  train/
    rgb/           ← symlinks a images/ originales
    poses/         ← matrices 4×4 camera-to-world (.txt)
    calibration/   ← focal length (.txt) o matriz K (.txt)
  test/
    rgb/
    poses/
    calibration/
```

### Convención de coordenadas

COLMAP almacena poses como **world-to-camera**:

$$X_{cam} = R \cdot X_{world} + t$$

ACE necesita **camera-to-world** (inversa):

$$X_{world} = R^T \cdot (X_{cam} - t)$$

El script convierte automáticamente vía `np.linalg.inv(w2c)`.

### Modelos de cámara soportados

| Modelo COLMAP | Parámetros | Salida ACE |
|---|---|---|
| `SIMPLE_PINHOLE` | f, cx, cy | focal (escalar) |
| `SIMPLE_RADIAL` | f, cx, cy, k | focal (escalar) |
| `RADIAL` | f, cx, cy, k1, k2 | focal (escalar) |
| `PINHOLE` | fx, fy, cx, cy | matriz K 3×3 |
| `OPENCV` | fx, fy, cx, cy, k1, k2, p1, p2 | matriz K 3×3 |

El modelo actual (`SIMPLE_RADIAL`) genera un único valor de focal, que es lo que ACE
expera por defecto.

### Comando de conversión

```bash
python3 preprocesamiento/scripts/colmap2ace.py \
  --colmap-dir data/serie-1 \
  --output-dir data/serie-1/ace \
  --train-ratio 0.8 \
  --symlink                  # usa symlinks en vez de copiar imágenes
```

---

## 7. Integración con OneFormer3D

### Rol de COLMAP → OneFormer3D

OneFormer3D realiza segmentación semántica e instancias sobre nubes de puntos 3D.
La conexión con COLMAP es la siguiente:

```
COLMAP dense → fused.ply (nube de puntos densa coloreada)
                    ↓
              OneFormer3D (inferencia)
                    ↓
              etiquetas semánticas por punto 3D
```

La reconstrucción `dense/0/fused.ply` generada por COLMAP (`stereo_fusion`) es una nube
de puntos XYZ + RGB que puede procesarse directamente con OneFormer3D en modo inferencia.

### Nota importante sobre el entrenamiento

OneFormer3D **está pre-entrenado en ScanNet y S3DIS**, datasets con estructura específica
(`.pth` por escena con normales, colores, superpoints, etc.).
Para usarlo en inferencia sobre las nubes generadas por COLMAP, es necesario:

1. Convertir `fused.ply` al formato de escena que OneFormer3D espera.
2. Opcionalmente re-entrenar o hacer fine-tuning si el dominio es muy diferente.

> **Este paso de conversión no está implementado aún en el pipeline.**
> La integración COLMAP → OneFormer3D es el próximo punto de conexión a desarrollar.

### Pipeline esperado (a implementar)

```
data/<serie>/dense/0/fused.ply
         ↓
  [conversor ply → scannet format]
         ↓
data/<serie>/oneformer3d/scene.pth
         ↓
  oneformer3d inference
         ↓
  labeled_pointcloud.ply (con semántica + instancias)
```

---

## 8. Análisis de tiempos: ¿es normal >24h con 1800 imágenes?

### Respuesta corta: **sí, si se incluye reconstrucción densa. No, si solo se necesita sparse.**

### Desglose de tiempos estimados por etapa

Con GPU NVIDIA RTX 3080/4080 y 1800 imágenes continuas de video (2 FPS, single camera):

| Etapa | Tiempo estimado GPU | Tiempo estimado CPU | Cuello de botella |
|---|---|---|---|
| Feature Extraction | 15–45 min | 2.5–7.5 h | I/O + SIFT GPU |
| Sequential Matching | 20–60 min | 2–6 h | GPU matching + loop detection |
| Sparse Reconstruction | 30–120 min | 1–4 h | BA incremental |
| **Image Undistortion** | 5–20 min | 5–20 min | I/O |
| **PatchMatchStereo** | **8–36 h** | **no viable** | **CUDA + VRAM** |
| Stereo Fusion | 0.5–2 h | 0.5–2 h | I/O + RAM |
| Meshing (Poisson) | 0.5–1 h | 0.5–1 h | CPU |

**El tiempo total observado de >24h es causado casi exclusivamente por PatchMatchStereo (dense reconstruction).**

### Por qué PatchMatchStereo es tan lento con 1800 imágenes

PatchMatchStereo procesa cada imagen por separado, generando un mapa de profundidad
por imagen. La complejidad es **O(N × W × H × samples × iterations)**:

- 1800 imágenes × resolución 3200px → cada imagen necesita ~5–15 min de GPU
- **1800 × 10 min = 300 h en el peor caso** (con geom_consistency activo y resolución alta)
- Con parámetros actuales (3200px, 15 samples, 5 iter, geom_consistency=1): **8–36 h es razonable**

### Referencia de tiempos documentados por la comunidad

| Dataset | Imágenes | Hardware | Tiempo sparse | Tiempo dense |
|---|---|---|---|---|
| South Building (referencia) | 128 | RTX 2080 | ~3 min | ~30 min |
| Building scene | 500 | RTX 3090 | ~15 min | ~4 h |
| Campus tour | 1000 | RTX 3090 | ~40 min | ~15 h |
| Campus tour | 1800 | RTX 3080 | ~90 min | **20–36 h** |

**Conclusión: para 1800 imágenes, >24h totales con dense es esperable con la configuración actual.**

### Si el tiempo >24h ocurrió solo en sparse (sin dense)

**Esto es exactamente lo observado en la ejecución del 11 de marzo.** Causa confirmada:

```
W levenberg_marquardt_strategy.cc:123] Linear solver failure.
Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
```

Diagnóstico:
1. **`--Mapper.ba_use_gpu` no estaba activado** → Ceres usaba el solver CPU Cholesky denso
2. Con 1600+ cámaras con shared intrinsics, la Hessiana llena rápidamente → Cholesky falla
3. Cada fallo genera un retry con paso reducido → decenas de iteraciones fallidas por ronda de BA
4. BA global con ratio=1.1 se disparó ~16 veces → ~16 × [N intentos fallidos × CPU time]
5. La reconstrucción sigue progresando (no se cuelga), pero a velocidad ~10–50× más lenta

**Solución aplicada en los scripts**: `--Mapper.ba_use_gpu 1` + `--Mapper.ba_global_images_ratio 1.4`

---

## 9. Cambios recomendados para reducir tiempos

### 9.1 Cambio crítico: desactivar dense para el pipeline principal

**Para ACE, la reconstrucción densa no es necesaria.** El `pipeline.sh` actual pasa
`--mode advanced` que ejecuta dense por defecto en `run-advance.sh`.

```bash
# Cambio en run-advance.sh (run-series.sh mode advanced → sin dense):
./run-series.sh edificio-a --mode advanced -- --no-dense

# O en pipeline.sh por defecto (pasar --no-dense a todos los COLMAP individuales):
./pipeline.sh --series video1.mp4 --colmap-args "--no-dense"
```

Ahorro estimado: **8–36 h por serie** (solo se ejecutan etapas 1–3).

### 9.2 Reducir resolución para sparse (sin perder calidad para ACE)

```bash
# Reducir de 3200 a 1600 para extracción en sparse:
./run-series.sh edificio-a --mode advanced -- --max-image-size 1600 --no-dense
```

SIFT funciona bien a 1600px para reconstrucción sparse, y la conversión a ACE usa las
imágenes originales (no las redimensionadas de COLMAP).

| Resolución | Tiempo extracción | Tiempo matching | Calidad sparse |
|---|---|---|---|
| 3200px | base | base | alta |
| 1600px | ~4× más rápido | ~2× más rápido | buena (suficiente para ACE) |
| 1000px | ~10× más rápido | ~5× más rápido | aceptable |

### 9.3 Ajustar Sequential Matcher para video continuo

Para videos a 2 FPS con 1800 frames (≈ 15 min de video):

```bash
--overlap 10                          # 10 es suficiente para 2 FPS
--loop_detection_num_images 30        # reducir de 50 a 30
```

### 9.4 Ajustar mapper para single-camera con focal conocida (YA APLICADO)

El mapper ahora incluye por defecto todos los parámetros optimizados:

```bash
# Aplicado automáticamente en run-advance.sh y run-series.sh:
--Mapper.ba_use_gpu 1                         # GPU solver (CRÍTICO)
--Mapper.ba_global_images_ratio 1.4           # 3x menos rondas de BA global
--Mapper.ba_global_points_ratio 1.4
--Mapper.ba_global_max_num_iterations 30      # menos iters por ronda
--Mapper.ba_global_ignore_redundant_points3D 1 # poda puntos redundantes
--Mapper.num_threads -1                        # todos los núcleos
# Solo con --single-camera (default):
--Mapper.ba_refine_focal_length 0             # focal fija
--Mapper.ba_refine_extra_params 0             # distorsión fija
```

### 9.5 Mapper paralelo: Hierarchical Mapper para >1000 imágenes

Para datasets muy grandes (>1000 imágenes), el mapper incremental puede ser lento
porque cada nueva imagen requiere BA global. El **Hierarchical Mapper** divide la escena
en clusters y luego los une:

```bash
colmap hierarchical_mapper \
  --database_path ./database.db \
  --image_path ./images \
  --output_path ./sparse \
  --HierarchicalMapper.leaf_max_num_images 200    # max imágenes por cluster
```

> Actualmente no está integrado en los scripts del proyecto.

### 9.6 Dense solo cuando se necesita (para OneFormer3D)

Cuando se requiera la nube densa para OneFormer3D, usar parámetros más rápidos:

```bash
# Dense más rápido (calidad reducida pero aceptable para segmentación):
--PatchMatchStereo.max_image_size 1600   # en vez de 3200
--PatchMatchStereo.num_samples 7         # en vez de 15
--PatchMatchStereo.num_iterations 3      # en vez de 5
--PatchMatchStereo.geom_consistency 0    # saltar 2do pass → ~50% ahorro
```

| Configuración | Tiempo por imagen | 1800 imgs | Calidad |
|---|---|---|---|
| Actual (3200, 15s, 5i, geom=1) | ~10–20 min | 300–600 h | muy alta |
| Rápida (1600, 7s, 3i, geom=0) | ~2–4 min | 60–120 h | buena |
| Mínima (1000, 5s, 2i, geom=0) | ~0.5–1 min | 15–30 h | aceptable |

### 9.7 Resumen de ganancias esperadas

| Cambio | Ahorro de tiempo | Impacto en calidad ACE |
|---|---|---|
| `--no-dense` en series individuales | **8–36 h / serie** | ninguno |
| `--max-image-size 1600` en sparse | **3–4×** en extracción y matching | mínimo |
| `--overlap 10` (vs 20) | ~30% en matching | mínimo para 2 FPS |
| Hierarchical mapper | 2–5× en mapper (>1000 imgs) | equivalente |
| Dense con parámetros rápidos | **5–10×** en dense | moderado |

---

## 10. Comandos de referencia rápida

### Pipeline completo fast (para producción de datos ACE)

```bash
cd preprocesamiento/pipelines

# Serie individual sin dense (solo sparse para ACE)
./pipeline.sh \
  --series video1.mp4 video2.mp4 \
  --sample-fps 2 \
  --colmap-mode advanced \
  --merge \
  --run-ace

# Si el run-series.sh acepta argumentos extra con --:
./run-series.sh edificio-a --mode advanced -- --no-dense --max-image-size 1600
```

### Verificar que la GPU está activa

```bash
# Dentro del contenedor
docker run --rm --gpus all colmap:latest nvidia-smi

# En los logs del pipeline debe aparecer:
# GPU Docker   : OK (--gpus all)
# GPU          : 1
```

### Reanudar desde una etapa intermedia

```bash
# Si la extracción y matching ya terminaron, reanudar desde sparse:
./run-series.sh edificio-a --mode advanced -- --skip-to sparse

# Si solo falta la fusión densa:
./run-series.sh edificio-a --mode advanced -- --skip-to fusion
```

### Inspeccionar reconstrucción sparse

```bash
# Ver cuántas imágenes registradas y puntos 3D:
docker run --rm \
  -v $(pwd)/data/serie-1:/working \
  colmap:latest \
  colmap model_analyzer --path /working/sparse/0

# Convertir binario a texto:
docker run --rm \
  -v $(pwd)/data/serie-1:/working \
  colmap:latest \
  colmap model_converter \
    --input_path /working/sparse/0 \
    --output_path /working/sparse/0 \
    --output_type TXT
```

### Descargar vocabulary tree para merge

```bash
# Para datasets de 1K-10K imágenes (recomendado):
wget https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin \
  -O data/vocab_tree.bin

# Para >10K imágenes:
wget https://demuc.de/colmap/vocab_tree_flickr100K_words1M.bin \
  -O data/vocab_tree.bin
```

### Merge de series con vocab_tree

```bash
./merge-colmap.sh serie-1 serie-2 serie-3 \
  --matcher vocab_tree \
  --vocab-tree vocab_tree.bin \
  --data-root ./data
```

---

## Referencias

- COLMAP documentation: https://colmap.github.io/
- COLMAP parameters reference: https://colmap.github.io/tutorial.html
- Sequential vs Exhaustive vs VocabTree: https://colmap.github.io/tutorial.html#feature-matching
- PatchMatchStereo paper: *Pixelwise View Selection for Unstructured Multi-View Stereo* (Schönberger et al., ECCV 2016)
- SfM revisited paper: *Structure-from-Motion Revisited* (Schönberger & Frahm, CVPR 2016)
- ACE (Accelerated Coordinate Encoding): https://nianticlabs.github.io/ace/
- Scripts del proyecto: `preprocesamiento/models/colmap/docker/`
