# Diagrama de arquitectura de UniWhere

Este documento representa la arquitectura lógica de UniWhere en formato visual, con énfasis en etapas funcionales y relación entre modelos. El objetivo es describir cómo fluye la información dentro del sistema, no detallar implementación.

## Vista 1. Flujo lógico del sistema

```mermaid
flowchart TD
    A[Captura del entorno<br/>video, imágenes, recorridos] --> B[Preparación de observaciones<br/>frames y secuencias normalizadas]
    B --> C[Reconstrucción geométrica 3D<br/>modelo espacial del campus]
    C --> D[Relocalización visual<br/>estimación de pose del usuario]
    C --> E[Comprensión semántica 3D<br/>interpretación del entorno]
    D --> F[Modelo espacial navegable<br/>geometría + pose + semántica]
    E --> F
    F --> G[Cálculo de rutas y asistencia espacial]
    G --> H[Visualización en realidad aumentada]

    B -. herramienta .-> B1[VideoFrameExtractor]
    C -. modelos .-> C1[COLMAP]
    C -. modelos .-> C2[SLAM3R]
    D -. modelo .-> D1[ACE]
    E -. modelo .-> E1[OneFormer3D]
```

## Interpretación de la vista de flujo

La secuencia lógica del sistema puede leerse así:

1. El campus se observa mediante video o imágenes.
2. Las observaciones se preparan para alimentar el pipeline visual.
3. Se construye un modelo geométrico 3D del entorno.
4. El usuario se relocaliza dentro de ese modelo.
5. El entorno se enriquece con significado semántico.
6. Se consolida un modelo espacial navegable.
7. Se calcula una guía hacia el destino.
8. La guía se presenta con AR.

## Vista 2. Arquitectura por capas

```mermaid
flowchart TB
    subgraph L1[CAPA 1. Captura y preparación]
        L1A[Adquisición del entorno]
        L1B[Selección y preparación de frames]
        L1C[Normalización de observaciones]
    end

    subgraph L2[CAPA 2. Modelado espacial]
        L2A[Reconstrucción 3D offline]
        L2B[Reconstrucción 3D densa o continua]
        L2C[Nube de puntos o representación espacial]
    end

    subgraph L3[CAPA 3. Localización y comprensión]
        L3A[Relocalización visual del usuario]
        L3B[Estimación de pose 6DoF]
        L3C[Segmentación y semántica 3D]
        L3D[Modelo espacial navegable]
    end

    subgraph L4[CAPA 4. Navegación y presentación]
        L4A[Cálculo de rutas]
        L4B[Asistencia contextual]
        L4C[Interfaz de realidad aumentada]
    end

    L1 --> L2 --> L3 --> L4

    M1[VideoFrameExtractor] --- L1
    M2[COLMAP] --- L2A
    M3[SLAM3R] --- L2B
    M4[ACE] --- L3A
    M5[OneFormer3D] --- L3C
```

## Interpretación de la vista por capas

### Capa 1. Captura y preparación

Convierte el entorno físico en datos visuales utilizables. Aquí el sistema todavía no navega ni localiza; solo prepara la evidencia necesaria para que los modelos posteriores trabajen con consistencia.

### Capa 2. Modelado espacial

Genera la representación tridimensional del campus. Esta capa construye la base geométrica que hace posible localizar al usuario y entender el entorno como espacio navegable.

### Capa 3. Localización y comprensión

Relaciona la observación actual del usuario con el modelo del entorno y agrega significado semántico al espacio. Esta es la capa donde la percepción se vuelve conocimiento operativo.

### Capa 4. Navegación y presentación

Transforma el conocimiento espacial en guía útil para el usuario. Es la capa que materializa la experiencia de navegación asistida en AR.

## Correspondencia con el repositorio

La estructura actual del repositorio se puede leer de forma consistente con estas capas:

- preprocesamiento concentra sobre todo las capas 1 y 2, y parte de la 3;
- backend debería consolidar la lógica operativa de localización, seguimiento y navegación;
- client representa la futura experiencia de usuario;
- docs conserva la documentación conceptual, académica y arquitectónica.

## Uso recomendado del diagrama

Este documento sirve como base para:

- explicar el proyecto en presentaciones,
- documentar el sistema en términos arquitectónicos,
- alinear el desarrollo con el entregable 1,
- y separar claramente la lógica del sistema de los detalles de implementación.