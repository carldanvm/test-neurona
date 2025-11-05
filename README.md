# Sistema de Detección de Neumonía - Red Neuronal

## 📋 Descripción
Proyecto universitario que implementa una red neuronal simple para detectar posibles casos de neumonía en radiografías de tórax. Desarrollado con fines académicos para demostrar conceptos fundamentales de machine learning y procesamiento de imágenes.

## 🎯 Objetivo Académico
Crear un sistema de clasificación binaria que pueda distinguir entre radiografías normales y con neumonía, implementando desde cero una red neuronal con una capa oculta usando únicamente NumPy.

## 🏗️ Arquitectura del Sistema
- **Entrada:** 40,000 valores (imágenes 200×200 píxeles aplanadas)
- **Capa oculta:** 60 neuronas con activación sigmoid
- **Salida:** 1 neurona con activación sigmoid (probabilidad de neumonía)

## 📁 Estructura del Proyecto
```
test-neurona/
├── config.py              # Configuración centralizada
├── ImageProcessor.py      # Procesamiento de imágenes
├── Neurona.py            # Implementación de la red neuronal
├── main.py               # Interfaz de usuario
├── requirements.txt      # Dependencias
├── INFORME.md           # Informe técnico del proyecto
├── README.md            # Este archivo
├── neurona/             # Modelo entrenado (se genera automáticamente)
├── imagenes-para-entrenar/     # Imágenes procesadas para entrenamiento
├── imagenes-para-testing/      # Imágenes para evaluación
└── imagenes-para-entrenar-sin-procesar/      # Imágenes originales
```

## 🛠️ Tecnologías Utilizadas
- **Python 3.x**
- **NumPy** - Operaciones matriciales y cálculos matemáticos
- **PIL (Pillow)** - Procesamiento de imágenes
- **JSON** - Persistencia del modelo

## 📦 Instalación
1. Clona o descarga el proyecto
2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 🚀 Uso
Ejecuta el programa principal:
```bash
python main.py
```

### Opciones del menú:
1. **Procesar imágenes** - Redimensiona y convierte imágenes a escala de grises
2. **Entrenar neurona** - Entrena el modelo con las imágenes procesadas
3. **Hacer predicción** - Clasifica nuevas imágenes (modo continuo)
4. **Salir** - Termina el programa

## ⚙️ Configuración
Todos los parámetros se pueden modificar en `config.py`:
- Tamaño de imagen: 200×200 píxeles
- Tasa de aprendizaje: 0.001
- Épocas por defecto: 300
- Neuronas capa oculta: 60
- Umbral de predicción: 0.5

## 📊 Dataset y Resultados

### Dataset Utilizado:
**Chest X-Ray Images (Pneumonia)**
- **Fuente:** [Kaggle - Chest X-Ray Images (Pneumonia)](https://kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)
- **Descripción:** Radiografías de tórax clasificadas en casos normales y con neumonía
- **Entrenamiento:** 2,682 imágenes balanceadas (1,341 normales + 1,341 neumonía)
- **Prueba:** 470 imágenes (235 por cada clase)

### Resultados Obtenidos:
- **Precisión general:** 80.21%
- **Precisión en casos normales:** 78.72%
- **Precisión en casos de neumonía:** 81.70%

## 🔬 Características Técnicas
- Implementación desde cero sin frameworks de ML
- Inicialización de pesos Xavier/He
- Algoritmo de backpropagation
- Balanceo automático del dataset
- Evaluación en datos no vistos durante entrenamiento
- Guardado/carga automática del modelo

## 📚 Propósito Educativo
Este proyecto fue desarrollado para:
- Comprender los fundamentos de las redes neuronales
- Implementar algoritmos de machine learning desde cero
- Practicar procesamiento de imágenes médicas
- Aplicar conceptos de clasificación binaria
- Evaluar modelos con métricas apropiadas

## ⚠️ Limitaciones
- **Solo para fines académicos** - No usar para diagnósticos reales
- Arquitectura simple (una sola capa oculta)
- Dataset limitado
- Sin técnicas avanzadas como data augmentation o CNN

## 🔮 Posibles Mejoras
- Implementar redes convolucionales (CNN)
- Aumentar el tamaño del dataset
- Agregar más capas ocultas
- Implementar técnicas de regularización
- Añadir validación cruzada

## 📄 Licencia
Este proyecto es de uso académico únicamente.

---
**Nota:** Este sistema es un proyecto educativo y no debe utilizarse para diagnósticos médicos reales.