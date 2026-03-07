# Entregable 1

## Desarrollo de un sistema inteligente de navegación indoor mediante realidad aumentada y nubes de puntos para campus universitarios

Este documento es una conversión a markdown, limpia y legible, del contenido de [docs/Entregable 1-1.pdf](docs/Entregable%201-1.pdf).

## Resumen ejecutivo

El proyecto UniWhere propone un sistema de wayfinding inteligente para entornos universitarios complejos, especialmente campus con edificios, pasillos, pisos y zonas interiores donde el GPS no ofrece precisión suficiente.

La idea central es combinar:

- realidad aumentada en dispositivos móviles,
- localización visual mediante SLAM,
- reconstrucción y procesamiento de nubes de puntos 3D,
- y cálculo de rutas hacia puntos de interés.

El objetivo es que un usuario pueda orientarse y desplazarse dentro del campus con ayuda visual contextualizada, sin depender del GPS en interiores.

## Problema

La navegación en espacios interiores presenta limitaciones técnicas que no están resueltas por sistemas tradicionales como GPS:

- el posicionamiento en interiores es impreciso,
- no existe buena resolución vertical para distinguir pisos o niveles,
- los entornos complejos generan desorientación,
- la señalización estática no adapta rutas al contexto del usuario,
- y los cambios de iluminación, oclusiones y multitudes afectan la localización visual.

En campus universitarios esto impacta directamente la capacidad de encontrar laboratorios, oficinas, aulas y otros puntos de interés.

## Justificación

### Justificación técnica

Los avances recientes en visión por computador permiten construir una solución factible basada en:

- SLAM visual para localización y mapeo simultáneo,
- reconstrucción 3D con video monocular o sensores de profundidad,
- nubes de puntos densas para representar el entorno,
- procesamiento acelerado por GPU,
- y modelos de reconocimiento de lugares para relocalización precisa.

### Justificación desde HCI

La realidad aumentada puede reducir la carga cognitiva del usuario porque muestra la guía directamente sobre el entorno físico. En lugar de interpretar mapas abstractos, el usuario sigue indicaciones visuales ancladas al espacio real.

### Justificación académica y aplicada

El proyecto integra áreas activas de investigación:

- navegación indoor,
- realidad aumentada móvil,
- visual localization,
- reconstrucción 3D,
- procesamiento de nubes de puntos,
- e interacción humano-computadora.

También tiene valor aplicado inmediato para campus inteligentes y otras infraestructuras complejas como hospitales o centros logísticos.

## Objetivo general

Desarrollar un sistema inteligente de navegación indoor mediante realidad aumentada y procesamiento de nubes de puntos que permita localizar y guiar usuarios hacia puntos de interés dentro de campus universitarios, usando SLAM visual y visión por computador sin depender de infraestructura GPS.

## Objetivos específicos

1. Identificar conceptos y estado del arte sobre navegación indoor, AR, SLAM visual y nubes de puntos.
2. Diseñar la arquitectura del sistema, incluyendo captura, reconstrucción, localización, cálculo de rutas y visualización AR.
3. Implementar un prototipo funcional integrando reconstrucción 3D, localización visual y renderizado de ayudas de navegación.
4. Validar el prototipo con métricas de precisión, rendimiento y usabilidad en un entorno universitario controlado.

## Metodología de investigación

El entregable describe una metodología longitudinal dividida en cuatro fases.

### Fase 1. Revisión sistemática de la literatura

Se realiza una revisión siguiendo PRISMA para estudiar:

- indoor navigation,
- augmented reality,
- point clouds,
- SLAM,
- visual positioning.

Se consultan bases como ACM Digital Library, IEEE Xplore, Web of Science y Springer Link.

### Fase 2. Abstracción, modelado y diseño

Se define la arquitectura general del sistema, incluyendo:

- captura de datos,
- procesamiento de nubes de puntos,
- localización visual en tiempo real,
- cálculo de rutas,
- y visualización móvil con AR.

### Fase 3. Prototipado

Se implementa el prototipo funcional integrando los módulos técnicos de reconstrucción, localización y visualización.

### Fase 4. Validación

Se evalúa el sistema con usuarios reales y métricas de:

- precisión de localización,
- desempeño computacional,
- efectividad de navegación,
- y usabilidad.

## Marco teórico sintetizado

El documento sustenta la solución con un conjunto de tecnologías y líneas de trabajo.

### 1. Realidad aumentada móvil

La AR sirve como capa de interacción para mostrar instrucciones de navegación sobre el entorno real. Esto mejora la comprensión espacial y reduce la dependencia de mapas tradicionales.

### 2. SLAM visual

SLAM permite estimar la pose del dispositivo y construir un mapa del entorno en tiempo real. Es la base para saber dónde está el usuario y para anclar correctamente elementos virtuales.

### 3. Reconstrucción 3D y nubes de puntos

Las nubes de puntos ofrecen una representación geométrica del entorno. Sirven para modelar edificios, entender la estructura espacial y soportar tareas de localización, segmentación y navegación.

### 4. Reconocimiento de lugares y relocalización

El sistema necesita comparar observaciones actuales con mapas o escenas previamente registradas para recuperar la posición del usuario cuando hay cambios de vista, iluminación o movimiento.

### 5. Segmentación y comprensión semántica

Para una navegación más útil, no basta con reconstruir geometría. También es importante identificar objetos, áreas y regiones del entorno para enriquecer el mapa con significado.

## Arquitectura conceptual esperada

El entregable plantea una arquitectura con estos bloques principales:

1. Captura de video o imágenes del entorno.
2. Extracción y preparación de frames.
3. Reconstrucción 3D y generación de nube de puntos.
4. Localización visual y estimación de pose del usuario.
5. Procesamiento semántico del entorno.
6. Cálculo de rutas hacia puntos de interés.
7. Visualización en realidad aumentada sobre dispositivo móvil.

## Lectura práctica del alcance

Visto desde ingeniería, el proyecto no es solo una app móvil. Es un pipeline completo de percepción espacial para campus:

- primero se captura el entorno,
- luego se construye un modelo 3D navegable,
- después se aprende o calcula cómo relocalizar al usuario dentro de ese modelo,
- y finalmente se renderiza la guía en AR.

## Estado del repositorio frente al entregable

La estructura actual del repositorio coincide con esa visión por etapas:

- preprocesamiento concentra captura, reconstrucción, relocalización y visualización técnica,
- backend contiene módulos vinculados al seguimiento y futura localización,
- client existe como espacio reservado para una futura interfaz o aplicación cliente,
- docs conserva entregables y material de referencia.

En el estado actual, gran parte del trabajo implementado está organizado como submódulos de investigación integrados al pipeline.

## Conclusión

El entregable 1 define a UniWhere como un sistema inteligente de navegación indoor para campus universitarios, apoyado en realidad aumentada, localización visual y nubes de puntos 3D. El valor principal del proyecto está en unir reconstrucción espacial, relocalización precisa y una interfaz AR comprensible para el usuario final.