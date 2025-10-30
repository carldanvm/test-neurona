# Guía de Uso - Interfaz Gráfica

## 🚀 Cómo Iniciar

### Interfaz Gráfica (por defecto)
```bash
python main.py
```

### Interfaz de Línea de Comandos (opcional)
```bash
python main.py --cli
```

## 📦 Instalación de Dependencias

```bash
pip install -r requirements.txt
```

**Nota:** `tkinterdnd2` es opcional. Si no está instalado, podrás seleccionar imágenes con un botón en lugar de arrastrar y soltar.

## 🖥️ Características de la GUI

### 1. **Procesar Imágenes**
- Botón: 📁 **Procesar Imágenes**
- Función: Preprocesa todas las imágenes del directorio `imagenes-sin-procesar`
- Las imágenes se redimensionan y convierten a escala de grises
- Se guardan en `imagenes-para-entrenar`

### 2. **Entrenar Neurona**
- Botón: 🧠 **Entrenar Neurona**
- Función: Entrena la red neuronal con las imágenes procesadas
- Permite configurar el número de épocas (recomendado: 300)
- Muestra progreso en tiempo real en el log
- Guarda el modelo entrenado automáticamente

### 3. **Hacer Predicción**
- **Cargar Imagen:**
  - 🖼️ Arrastra una imagen al área azul, o
  - Haz clic en el área azul para seleccionar una imagen
  
- **Analizar:**
  - Botón: 🔍 **Analizar Imagen**
  - Vista previa de la imagen en el lado izquierdo
  - Resultado con diagnóstico y confianza en el lado derecho

### 4. **Registro de Actividad**
- Panel inferior que muestra todas las operaciones
- Útil para seguir el progreso del entrenamiento
- Muestra mensajes de error si algo falla

## 🎯 Flujo de Trabajo Típico

1. **Primera vez:**
   - ✅ Coloca imágenes en `imagenes-sin-procesar/normal` y `imagenes-sin-procesar/neumonia`
   - ✅ Haz clic en "Procesar Imágenes"
   - ✅ Haz clic en "Entrenar Neurona" y configura las épocas
   - ✅ Espera a que termine el entrenamiento

2. **Hacer predicciones:**
   - ✅ Arrastra o selecciona una radiografía
   - ✅ Haz clic en "Analizar Imagen"
   - ✅ Ve el resultado instantáneamente

## 📊 Interpretación de Resultados

- **NORMAL** (Verde ✓): No se detectó neumonía
- **NEUMONIA** (Rojo ⚠️): Se detectó neumonía
- **Confianza**: Porcentaje de certeza del modelo (0-100%)

## 🔧 Solución de Problemas

### La GUI no inicia
- Verifica que Pillow esté instalado: `pip install Pillow`
- Usa la CLI en su lugar: `python main.py --cli`

### No puedo arrastrar imágenes
- Instala tkinterdnd2: `pip install tkinterdnd2`
- O usa el botón para seleccionar archivos

### Error al predecir
- Asegúrate de haber entrenado la neurona primero
- Verifica que la imagen sea válida (PNG, JPG, etc.)

## 💡 Consejos

- **Entrenamiento:** Más épocas = mejor precisión (pero más tiempo)
- **Imágenes:** Usa imágenes de radiografías de tórax
- **Balance:** El sistema balancea automáticamente las clases durante el entrenamiento
- **Modelo:** Se guarda automáticamente en la carpeta `neurona/`

## 🖼️ Formatos de Imagen Soportados
- PNG, JPG, JPEG
- BMP, GIF
- TIFF, TIF
- WEBP
