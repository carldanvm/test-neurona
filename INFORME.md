# Sistema de Detección de Neumonía usando Red Neuronal

## Objetivo del proyecto:
Desarrollar un sistema que mediante el uso de una red neuronal, pueda detectar posibles casos de neumonía en imágenes de radiografías de tórax.

## Metodología:
El sistema se compone de 3 procesos fundamentales:

### 1. Procesamiento de imágenes:
Las imágenes utilizadas para entrenar la red neuronal son radiografías de tórax, el dataset está clasificado en 2 carpetas: una para casos normales y otra para casos de neumonía. Dichas imágenes son redimensionadas a 200x200 píxeles y convertidas a escala de grises.

Para el entrenamiento y predicción, las imágenes son convertidas a arrays de NumPy y normalizadas a valores entre 0 y 1 (dividiendo por 255). Finalmente, las imágenes 2D se aplanan a un vector 1D de 40,000 elementos para alimentar la red neuronal.

### 2. Entrenamiento de la red neuronal:
La arquitectura de la red neuronal consiste en:
- **Capa de entrada:** 40,000 valores de entrada (200x200 píxeles aplanados)
- **Capa oculta:** 60 neuronas con función de activación sigmoid
- **Capa de salida:** 1 neurona con función sigmoid (probabilidad de neumonía)

**Parámetros de configuración:**
- Tasa de aprendizaje (learning rate): 0.001
- Número de épocas por defecto y recomendado: 300
- Umbral de predicción: 0.5
- Inicialización de pesos: Método Xavier/He

**Proceso de entrenamiento:**
1. **Balanceo del dataset:** Se equilibra automáticamente para tener igual cantidad de imágenes normales y con neumonía
2. **Inicialización:** Los pesos se inicializan aleatoriamente usando el método Xavier/He para evitar problemas de gradientes
3. **Forward propagation:** Los datos se propagan hacia adelante calculando las activaciones de cada capa
4. **Backward propagation:** Se calculan los gradientes del error y se propagan hacia atrás
5. **Actualización de parámetros:** Se ajustan los pesos y bias usando el algoritmo de descenso del gradiente
6. **Repetición:** El proceso se repite durante el número especificado de épocas
7. **Test de precisión:** Se evalúa el modelo con un conjunto de datos de prueba no usados en el entrenamiento para medir su precisión real
8. **Guardado del modelo:** Se guarda el modelo entrenado en un archivo JSON para su uso posterior

### 3. Predicción:
Para realizar predicciones en nuevas imágenes:
1. La imagen se procesa igual que en el entrenamiento (redimensionar, escala de grises, normalizar, aplanar)
2. Se carga el modelo previamente entrenado desde un archivo JSON
3. Se ejecuta forward propagation para obtener la probabilidad de salida
4. Si la probabilidad > 0.5 se clasifica como NEUMONÍA, caso contrario como NORMAL

## Implementación Técnica:

### Estructura del código:
- **config.py:** Configuración centralizada de todos los parámetros del sistema
- **ImageProcessor.py:** Clase encargada del procesamiento de imágenes
- **Neurona.py:** Clase encargada de la red neuronal y todos sus métodos
- **main.py:** Interfaz de usuario con menú interactivo

### Tecnologías utilizadas:
- **Python 3.x** como lenguaje principal
- **NumPy** para operaciones matriciales y cálculos matemáticos
- **PIL (Pillow)** para procesamiento de imágenes
- **JSON** para persistencia del modelo entrenado

## Resultados:

### Dataset utilizado:
- **Entrenamiento:** 2,682 imágenes balanceadas (1,341 normales + 1,341 neumonía)
- **Prueba:** 470 imágenes (235 por cada clase)

### Métricas de rendimiento:
- **Precisión general en test:** 80.21% (evaluado con imágenes no usadas en entrenamiento)
- **Precisión en casos normales:** 78.72% (185/235)
- **Precisión en casos de neumonía:** 81.70% (192/235)


## Conclusiones:
El sistema desarrollado alcanzó una precisión del 80.21% en la detección de neumonía con la siguiente configuración:
- Una sola capa oculta de 60 neuronas y una capa de salida de 1 neurona
- Imágenes de 200×200 píxeles
- Tasa de aprendizaje: 0.001
- Número de épocas: 300
- Umbral de predicción: 0.5
- Inicialización de pesos: Método Xavier/He

Aunque la precisión lograda es aceptable en términos académicos, para casos de uso reales se recomienda experimentar con diferentes parámetros y técnicas para mejorar el rendimiento. Por ejemplo:
- Añadir más capas ocultas
- Ampliar el dataset
- Migrar a una red neuronal convolucional (CNN)
