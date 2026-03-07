# UniWhere

UniWhere es una propuesta de navegación indoor para campus universitarios que combina reconstrucción 3D, localización visual, procesamiento semántico del entorno y visualización en realidad aumentada.

La finalidad del proyecto es permitir que un usuario pueda ubicarse dentro de espacios interiores complejos y recibir guía contextual hacia puntos de interés, incluso en lugares donde el GPS no ofrece precisión suficiente.

## Objetivo

El sistema busca resolver un problema de orientación espacial en infraestructuras complejas como campus, edificios multinivel, laboratorios o corredores interiores. En estos escenarios, la navegación tradicional falla por tres motivos principales:

- el GPS no ofrece precisión confiable en interiores,
- los mapas 2D no representan bien la experiencia espacial real,
- y la señalización física no se adapta al contexto del usuario.

UniWhere aborda ese problema mediante una arquitectura de percepción espacial que integra modelos de reconstrucción, relocalización y comprensión del entorno.

## Visión lógica del sistema

El repositorio representa un pipeline de navegación visual compuesto por etapas lógicas.

### 1. Captura del entorno

El sistema parte de videos o colecciones de imágenes capturadas en el campus. Esa información es la materia prima para construir el modelo espacial del entorno.

### 2. Preparación de observaciones visuales

Antes de reconstruir o localizar, el flujo organiza y normaliza la evidencia visual en forma de frames o secuencias. Esta etapa permite alimentar de forma consistente los modelos posteriores.

### 3. Reconstrucción espacial

En esta fase se genera una representación tridimensional del entorno. El objetivo no es solo producir una nube de puntos, sino disponer de una base geométrica navegable sobre la cual luego se pueda localizar al usuario y enriquecer el espacio con semántica.

### 4. Relocalización visual

Una vez existe un modelo del entorno, el sistema necesita estimar la pose del usuario o del dispositivo dentro de ese espacio. Esta etapa responde la pregunta central de navegación: dónde estoy y cómo se relaciona mi vista actual con el mapa construido.

### 5. Comprensión semántica del espacio

La navegación no depende únicamente de geometría. También requiere distinguir regiones, objetos o estructuras del entorno para que el mapa sea interpretable y útil para tareas de guiado.

### 6. Cálculo de guía

Con el usuario localizado y el entorno modelado, el sistema puede estimar rutas o secuencias de navegación hacia destinos relevantes dentro del campus.

### 7. Presentación en realidad aumentada

La salida final esperada es una experiencia de guiado visual en AR, donde la información se superpone al entorno real para reducir la carga cognitiva del usuario.

## Papel de los modelos integrados

La estructura actual del repositorio muestra una integración de modelos y herramientas especializadas, cada uno con un rol lógico dentro del sistema.

### Reconstrucción 3D

- SLAM3R se alinea con la necesidad de reconstrucción densa del entorno a partir de video RGB monocular en tiempo real.
- COLMAP aporta un enfoque clásico de Structure from Motion y Multi-View Stereo para reconstrucción geométrica desde imágenes.

Ambos encajan en la capa de modelado espacial del campus.

### Relocalización y posicionamiento visual

- ACE encaja en el problema de relocalización 6DoF a partir de imágenes RGB.

Su papel lógico dentro de UniWhere es permitir que una observación nueva del usuario se proyecte sobre un mapa ya aprendido o reconstruido.

### Comprensión semántica de la nube de puntos

- OneFormer3D encaja en la interpretación del entorno 3D mediante segmentación semántica, de instancias o panóptica.

Esto permite pasar de una geometría cruda a una representación con significado espacial.

### Preparación de datos

- VideoFrameExtractor cumple la función de transformar video en una secuencia utilizable para reconstrucción, análisis o entrenamiento.

## Estructura actual del repositorio

### docs

Contiene entregables académicos y documentación de apoyo del proyecto.

### preprocesamiento

Es el bloque más desarrollado actualmente. Reúne herramientas y modelos que soportan:

- extracción de frames,
- reconstrucción 3D,
- relocalización visual,
- y visualización técnica de resultados.

Desde una perspectiva lógica, aquí vive la mayor parte del pipeline de percepción espacial.

### backend

Representa la capa donde deberían consolidarse las capacidades operativas del sistema de navegación, especialmente seguimiento, localización y servicios asociados. Hoy su contenido visible está más concentrado en el bloque de seguimiento 3D que en una aplicación backend completa.

### client

Actúa como espacio reservado para la futura experiencia cliente, previsiblemente una app o interfaz de navegación. En el estado actual no concentra la lógica principal del proyecto.

## Estado actual

El repositorio todavía se comporta más como una plataforma de integración de modelos de percepción espacial que como una aplicación final cerrada. Eso es consistente con el entregable 1: primero se está consolidando la base de reconstrucción, relocalización y comprensión del entorno; después podrá cerrarse la capa de guiado y experiencia AR de usuario.

## Instalación de herramientas auxiliares

El repositorio incluye dos instaladores para preparar VideoFrameExtractor y CloudCompare automáticamente.

### Linux o macOS con bash

Ejecuta:

```bash
./scripts/install_tools.sh
```

Qué hace:

- instala VideoFrameExtractor desde [preprocesamiento/VideoFrameExtractor](preprocesamiento/VideoFrameExtractor);
- si existe uv, lo instala como tool persistente con `uv tool install --editable`;
- si uv no existe, crea un entorno virtual local en `.tools/videoframeextractor` usando Python 3.12+;
- instala CloudCompare con flatpak si está disponible;
- y, si no hay flatpak, intenta instalar CloudCompare con `apt-get`.

Nota sobre uv:

- cuando se usa uv, el comando esperado es `videoframeextractor` en el PATH del usuario;
- si no aparece inmediatamente en la sesión actual, abre una nueva terminal o ejecuta `uv tool update-shell`.

### Windows con PowerShell

Ejecuta:

```powershell
./scripts/install_tools.ps1
```

Qué hace:

- instala VideoFrameExtractor desde [preprocesamiento/VideoFrameExtractor](preprocesamiento/VideoFrameExtractor);
- si existe uv, lo instala como tool persistente con `uv tool install --editable`;
- si uv no existe, crea un entorno virtual local en `.tools/videoframeextractor` usando Python 3.12+;
- instala CloudCompare con `winget` usando el identificador oficial `CloudCompare.CloudCompare`.

### Requisitos mínimos

- para VideoFrameExtractor: `uv` o Python 3.12+;
- para CloudCompare en Linux: `flatpak` o `apt-get`;
- para CloudCompare en Windows: `winget`.

## Documentos relacionados

- [docs/Entregable-1-1.md](docs/Entregable-1-1.md): versión markdown del entregable 1.
- [docs/arquitectura-logica.md](docs/arquitectura-logica.md): flujo lógico del sistema y rol de los modelos integrados.
- [docs/diagrama-arquitectura.md](docs/diagrama-arquitectura.md): diagrama formal del sistema en vista de flujo y por capas.
- [docs/uso-colmap-docker.md](docs/uso-colmap-docker.md): flujo recomendado para correr COLMAP con Docker usando series en preprocesamiento/data.