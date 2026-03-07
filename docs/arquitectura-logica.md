# Arquitectura lógica de UniWhere

## Propósito

Este documento describe UniWhere desde una perspectiva de arquitectura lógica y de modelos. El foco está en qué problema resuelve cada bloque del sistema y cómo se conectan entre sí, no en detalles de implementación.

## Problema que resuelve el sistema

UniWhere busca resolver navegación indoor en campus universitarios complejos. El reto principal no es solo dibujar rutas, sino construir una representación espacial suficientemente rica para responder tres preguntas en tiempo real:

1. Cómo es el entorno.
2. Dónde está el usuario dentro de ese entorno.
3. Cómo guiarlo de forma comprensible hasta un destino.

## Flujo lógico de extremo a extremo

### 1. Adquisición del entorno

El sistema comienza con observaciones visuales del campus:

- video,
- imágenes,
- o secuencias capturadas durante recorridos.

Esta fase tiene como objetivo convertir el espacio físico en evidencia visual suficiente para modelarlo digitalmente.

### 2. Preparación de insumos visuales

Antes de usar modelos de reconstrucción o localización, el sistema organiza las capturas en una secuencia estable de observaciones. En esta fase se prioriza que los datos tengan una forma consistente para alimentar los modelos posteriores.

Rol lógico:

- convertir video en frames utilizables,
- controlar densidad temporal de las observaciones,
- y estandarizar el material visual del pipeline.

Modelo o herramienta representativa:

- VideoFrameExtractor.

### 3. Reconstrucción geométrica del campus

Aquí el sistema transforma observaciones 2D en una representación 3D del entorno. Esta es una fase fundacional porque establece el mapa espacial sobre el que luego operan la localización y la navegación.

Rol lógico:

- construir geometría del entorno,
- producir nubes de puntos o representaciones equivalentes,
- y disponer de una base espacial navegable.

Modelos representativos:

- COLMAP, como pipeline clásico de reconstrucción a partir de múltiples vistas.
- SLAM3R, como enfoque de reconstrucción densa en tiempo real desde video RGB monocular.

Interpretación dentro de UniWhere:

COLMAP sirve como motor sólido para reconstrucciones estructuradas offline, mientras que SLAM3R encaja mejor en una visión dinámica y moderna del mapeo continuo del entorno.

### 4. Relocalización visual del usuario

Una vez existe un mapa del campus, el sistema debe determinar la pose del usuario respecto a ese mapa. Este es el núcleo operativo de la navegación: sin relocalización, no hay forma de saber desde qué punto deben generarse instrucciones o anclajes AR.

Rol lógico:

- estimar posición y orientación del dispositivo,
- asociar una observación actual con una región del mapa,
- y reubicar al usuario incluso tras cambios de perspectiva.

Modelo representativo:

- ACE.

Interpretación dentro de UniWhere:

ACE funciona como el módulo que conecta la observación presente del usuario con el mundo 3D ya modelado.

### 5. Comprensión semántica del entorno

La geometría por sí sola no basta para construir una experiencia de wayfinding útil. El sistema necesita diferenciar qué partes del entorno corresponden a objetos, zonas, límites o estructuras relevantes.

Rol lógico:

- enriquecer el mapa con significado,
- distinguir elementos del entorno,
- y permitir una representación más interpretable para navegación y consulta.

Modelo representativo:

- OneFormer3D.

Interpretación dentro de UniWhere:

OneFormer3D aporta la transición desde un mapa 3D geométrico hacia un mapa 3D entendible.

### 6. Modelo espacial navegable

Después de reconstruir, localizar y segmentar, el sistema puede consolidar un modelo espacial navegable del campus. Este modelo no es solo una nube de puntos, sino una estructura funcional que integra:

- geometría,
- referencia posicional,
- y significado semántico.

Ese modelo es la verdadera base de UniWhere.

### 7. Cálculo de rutas y asistencia espacial

Con el usuario ya ubicado dentro del modelo espacial, el sistema puede calcular trayectorias o secuencias de guiado hacia destinos concretos.

Rol lógico:

- conectar origen y destino dentro del espacio modelado,
- traducir esa relación a instrucciones navegables,
- y actualizar la guía conforme cambia la pose del usuario.

En la estructura actual del repositorio, esta capa existe más como objetivo arquitectónico que como bloque plenamente visible y consolidado.

### 8. Visualización en realidad aumentada

La última etapa consiste en presentar la guía de manera intuitiva sobre el entorno físico. El sistema busca que la navegación no dependa de interpretar un mapa abstracto, sino de seguir señales ancladas al espacio real.

Rol lógico:

- reducir carga cognitiva,
- mejorar comprensión espacial,
- y ofrecer una interacción contextualizada en tiempo real.

## Lectura por capas

La arquitectura puede entenderse como cuatro capas principales.

### Capa 1. Captura y preparación

Convierte el entorno físico en insumos visuales estables.

### Capa 2. Modelado espacial

Construye la geometría y la representación 3D del campus.

### Capa 3. Localización y comprensión

Determina la pose del usuario y enriquece el entorno con semántica.

### Capa 4. Navegación y presentación

Calcula rutas y las comunica mediante AR.

## Correspondencia con la estructura del repositorio

### preprocesamiento

Concentra principalmente las capas 1 y 2, y parte de la 3:

- extracción de observaciones,
- reconstrucción,
- relocalización,
- y visualización técnica.

### backend

Representa el lugar natural para consolidar la capa 3 y parte de la 4, especialmente cuando la lógica de localización, seguimiento y servicios de navegación madure hacia una arquitectura de aplicación.

### client

Representa la futura interfaz del usuario, donde debería materializarse la experiencia final de guiado AR.

### docs

Documenta el marco conceptual, académico y arquitectónico del proyecto.

## Interpretación general del estado actual

UniWhere todavía está más cerca de una plataforma de percepción espacial para navegación que de una aplicación de navegación terminada. Eso no es una carencia accidental, sino una señal de que el proyecto está en una etapa donde primero se consolidan los modelos fundamentales:

- reconstrucción,
- relocalización,
- y comprensión semántica.

La navegación visual asistida por AR depende de que esas capas sean sólidas.

## Resumen final

Visto lógicamente, UniWhere es un sistema que sigue esta cadena:

1. observa el campus,
2. lo reconstruye en 3D,
3. entiende dónde está el usuario dentro de ese modelo,
4. enriquece el mapa con semántica,
5. y finalmente transforma ese conocimiento en guía espacial aumentada.

Ese es el sentido de los modelos presentes en el repositorio y la razón por la que la estructura actual está organizada alrededor de ellos.