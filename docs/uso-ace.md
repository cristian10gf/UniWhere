# Uso de ACE (Accelerated Coordinate Encoding) en UniWhere

Última actualización: 2026-03-15

---

## ¿Qué es ACE?

ACE (Accelerated Coordinate Encoding, Brachmann et al. 2023) es un sistema de **relocalización visual 6DoF**: dada una foto de entrada, predice la **pose completa de la cámara** (posición 3D + orientación) dentro de un mapa previamente entrenado.

Arquitectura:
- **Encoder compartido** (`ace_encoder_pretrained.pt`): backbone MobileNetV2 preentrenado, fijo, extrae descriptores visuales de 512 dimensiones.
- **Head de escena** (`<modelo>.pt`): red ligera (FC layers) que mapea descriptores → coordenadas 3D de la escena. Es la única parte entrenada por escena.
- **PnP+RANSAC** (`dsacstar`): estima la pose final a partir de las correspondencias 2D-3D predichas.

El sistema **no usa un mapa de puntos explícito** en inferencia — la escena está codificada en los pesos del head.

---

## Pipeline completo: reconstruccion compatible → ACE

```
video → frames → MASt3R/COLMAP → colmap2ace.py → ACE train → ACE test
                                        ↓
                             data/<serie>/ace/{train,test}/{rgb,poses,calibration}/
```

`colmap2ace.py` no depende del reconstructor original: consume un layout compatible COLMAP.

### Conversión a ACE (`colmap2ace.py`)

El script divide las imágenes COLMAP en train/test y convierte los formatos:

| Archivo COLMAP | Formato ACE |
|---|---|
| `images.bin` → pose W2C | `poses/<frame>.txt` → pose C2W (4×4, invertida) |
| `cameras.bin` → intrínsecos | `calibration/<frame>.txt` → focal length (1 valor) |
| `images/<frame>.jpg` | `rgb/<frame>.jpg` (copiado) |

Contrato minimo esperado por escena (`data/<serie>` o `data/_merged`):

```text
images/
sparse/0/{cameras,images,points3D}.{txt|bin}
```

Cuando la escena viene de MASt3R, ese contrato se exporta por el adaptador externo antes de ACE.

**Convención de pose en ACE**: cámara-a-mundo (C2W), inversa de COLMAP (W2C). `colmap2ace.py` aplica la inversión automáticamente.

**Formato de calibración**: un único número = focal length en píxeles. Con `CAMERA_MODEL=OPENCV` se usa `f = (fx + fy) / 2`.

### Estructura de directorios

```
data/<serie>/
├── images/           ← frames originales (COLMAP)
├── sparse/0/         ← reconstrucción COLMAP
└── ace/
    ├── train/
    │   ├── rgb/          ← imágenes de entrenamiento
    │   ├── poses/        ← <frame>.txt  (matriz 4×4 C2W)
    │   └── calibration/  ← <frame>.txt  (focal length en px)
    ├── test/
    │   ├── rgb/
    │   ├── poses/
    │   └── calibration/
    └── models/
        └── v<timestamp>.pt   ← modelo entrenado
```

---

## Comandos de uso: `run-ace.sh`

Todos los comandos se ejecutan desde `preprocesamiento/`. El script gestiona automáticamente:
- Conversión COLMAP → ACE si la escena no existe
- Detección y paso de GPU al contenedor Docker
- Selección del modelo más reciente cuando no se especifica uno
- Nombrado automático de archivos de resultados con timestamp

### Referencia de opciones

```
./run-ace.sh <comando> <serie> [modelo.pt] [opciones] [-- args_ace]

Comandos:
  train <serie> [salida.pt]   Entrenar. Sin salida → crea v<timestamp>.pt en ace/models/
  test  <serie> [modelo.pt]   Evaluar.  Sin modelo → usa el .pt más reciente en ace/models/

<serie> puede ser:
  block_g_1200        →  data/block_g_1200/ace
  block_g_1200/ace    →  idem

Opciones propias:
  --train-ratio R     Ratio train/test en colmap2ace (default: 0.8)
  --session NOMBRE    Sufijo para archivos de resultados del test (default: timestamp)
  --data-root PATH    Carpeta base de datos (default: preprocesamiento/data)
  --cpu               Forzar CPU (sin GPU)

Args extras para ACE (después de --):
  --num_head_blocks N
  --use_aug True/False
  --use_half True/False
  ... (ver train_ace.py --help)
```

### Ejemplos de uso

```bash
cd preprocesamiento/

# Entrenar con defaults (recomendado para producción)
./run-ace.sh train block_g_1200

# Entrenar con nombre de modelo explícito
./run-ace.sh train block_g_1200 ace/models/experimento_v2.pt

# Entrenar con args extra de ACE
./run-ace.sh train block_g_1200 -- --num_head_blocks 2

# Evaluar el modelo más reciente
./run-ace.sh test block_g_1200

# Evaluar un modelo concreto
./run-ace.sh test block_g_1200 ace/models/v1_produccion.pt

# Evaluar con nombre de sesión (para comparar runs)
./run-ace.sh test block_g_1200 --session comparativa_v2

# Evaluar modelo concreto con sesión nombrada
./run-ace.sh test block_g_1200 ace/models/v1_produccion.pt --session v1_final

# Flujo recomendado con pipeline (MASt3R por defecto)
cd preprocesamiento/pipelines
./pipeline.sh --series-names block_g_1200 --run-ace

# Fallback explicito a COLMAP con el mismo paso ACE downstream
./pipeline.sh --series-names block_g_1200 --run-ace --reconstructor colmap
```

### Salida del test

El script imprime al finalizar un resumen y rutas de los archivos:

```
=======================================================
 Resultados del test
  Median rot     : 0.28°
  Median trans   : 1.5 cm
  Inlier ratio   : 0.052
  Archivo poses  : data/block_g_1200/ace/models/poses_ace_<session>.txt
=======================================================
```

**Archivos generados** (siempre dentro de `data/<serie>/ace/models/`):

| Archivo | Contenido |
|---|---|
| `test_ace_<session>.txt` | `<median_rot_deg> <median_trans_cm> <inlier_ratio>` |
| `poses_ace_<session>.txt` | Por frame: `<nombre> <quat_xyzw> <trans_xyz> <rot_err_deg> <trans_err_cm> <inliers>` |

---

## Cómo funciona el entrenamiento de ACE

ACE no entrena en "épocas" convencionales. Internamente ejecuta un **curriculum de 3 pasos** con ~25.000 iteraciones de gradiente totales (~2-5 min en GPU RTX 4000 Ada para 1200 imágenes):

| Paso | Función |
|---|---|
| **Step 1** | Bootstrap — head aprende distribución aproximada de la escena |
| **Step 2** | Refinamiento — regresión completa con minería de muestras difíciles |
| **Step 3** | Fine-tuning — retroalimentación RANSAC-in-the-loop, produce el checkpoint final |

El `.pt` guardado corresponde siempre al checkpoint de Step 3 (modelo final deployable).

### Parámetros de entrenamiento

| Parámetro | Default ACE | Notas |
|---|---|---|
| `--num_head_blocks` | `1` | Default del paper para todos sus benchmarks |
| `--training_buffer_size` | `8000000` | Patches en buffer; reducir si OOM en RAM |
| `--samples_per_image` | `1024` | Patches extraídos por imagen al llenar buffer |
| `--batch_size` | `5120` | Debe ser múltiplo de 512 |
| `--epochs` | `16` | Pasadas sobre el buffer |
| `--use_aug` | `True` | Augmentación de imagen (rotación, escala) |
| `--use_half` | `True` | FP16 (requiere GPU con soporte fp16) |
| `--learning_rate_max` | `0.005` | Pico del scheduler 1-cycle |
| `--learning_rate_min` | `0.0005` | Mínimo del scheduler 1-cycle |

El entrenamiento con defaults completos tarda **~2.6 min** para `block_g_1200` (959 imágenes, RTX 4000 Ada 20GB).

---

## Métricas de evaluación

### Umbrales de calidad para navegación indoor

| Umbral | Uso práctico |
|---|---|
| **5cm / 5°** | Pose de alta precisión (AR overlay, robótica) |
| **25cm / 2°** | Navegación peatonal en interiores |
| **50cm / 5°** | Localización de zona/pasillo |
| **>1m / >10°** | Fallo de relocalización |

Para UniWhere el umbral operacional mínimo es **50cm / 5°**.

Benchmark de referencia (ACE paper, 7-Scenes indoor):
- Escenas ~1000 frames: recall >85% a 5cm/5°

---

## Modelos entrenados: `block_g_1200`

**Dataset**: 959 imágenes train, 240 test. Serie: pasillo bloque G, planta baja.

### `v1_produccion.pt` — modelo en producción (2026-03-15)

Entrenado con defaults completos de ACE (`use_aug=True`, `use_half=True`, buffer 8M, 16 épocas).
Tiempo de entrenamiento: **2.6 min** en RTX 4000 Ada.

| Métrica | Valor | Evaluación |
|---|---|---|
| Recall 10cm/5° | **92.1%** | Excelente |
| Recall 5cm/5° | **81.2%** | Excelente (supera umbral AR) |
| Recall 2cm/2° | **57.9%** | Bueno |
| Recall 1cm/1° | **37.5%** | Aceptable |
| Median rot. error | **0.3°** | Excelente |
| Median trans. error | **1.5 cm** | Excelente |

### `step3_repro_seed2089.pt` — modelo histórico (2026-03-13, NO usar)

Entrenado con parámetros severamente reducidos (buffer 65K, 2 épocas, sin augmentación).
Procesó ~1000× menos datos que el default → nunca convergió.

| Métrica | Valor | Evaluación |
|---|---|---|
| Recall 10cm/5° | 0.0% | FALLO |
| Median rot. error | 139.3° | FALLO (~aleatorio) |
| Median trans. error | 588.4 cm | FALLO |

---

## Notas técnicas

### Compatibilidad MASt3R / COLMAP para ACE

- ACE permanece sin cambios: el contrato de entrada lo mantiene `colmap2ace.py`.
- MASt3R y COLMAP convergen al mismo layout de `images/` + `sparse/0/...`.
- En validacion de no-regresion conviene comparar ambos reconstructores con el mismo `--train-ratio`.

### Detección de GPU

El entrypoint de `ace:latest` es `python` (no bash), por lo que la detección de GPU no puede usar `nvidia-smi` dentro del contenedor. El script `run.sh` verifica CUDA con:

1. `nvidia-smi` en el **host** → confirma que hay GPU
2. `docker run --gpus all ace:latest -c "import torch; print(torch.cuda.is_available())"` → confirma que Docker puede pasar la GPU al entorno Python de ACE

Si ambos pasan, se usa `--gpus all`. Si el segundo falla, se imprime una advertencia con instrucciones (`nvidia-container-toolkit`).

### Archivos de resultados

Los archivos de test siempre se guardan en `data/<serie>/ace/models/` junto al `.pt`. El parámetro `--session` añade un sufijo al nombre para poder conservar múltiples evaluaciones:

```bash
test_ace_20260315_012345.txt    ← timestamp automático
test_ace_comparativa_v2.txt     ← --session comparativa_v2
poses_ace_20260315_012345.txt
poses_ace_comparativa_v2.txt
```

### Visualizador ACE (Rerun)

```bash
cd preprocesamiento/
./run-ace-viewer.sh block_g_1200 \
    --test-poses data/block_g_1200/ace/models/poses_ace_<session>.txt
```
