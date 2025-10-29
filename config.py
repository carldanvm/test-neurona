class Config:
    TRAINING_IMAGES_DIR = "imagenes-para-entrenar" ## directorio donde se guardan las imágenes para entrenar, tamaño ajustado y en escala de grises
    RAW_TRAINING_IMAGES_DIR = "imagenes-sin-procesar" ## directorio donde se guardan las imágenes para entrenar originales
    TEST_IMAGES_DIR = "imagenes-para-testing" ## directorio donde se guardan las imágenes para testing originales
    SUBDIRECTORIES = ["normal", "neumonia"] ## subdirectorios de las carpetas de entrenamiento y testing
    MODEL_DIR = "neurona" ## directorio donde se guardan los pesos de la neurona
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp'] ## formatos de las imágenes soportados
    MAX_TEST_SAMPLES = 1000 ## número máximo de imágenes para testing usado para calcular la precisión real

    IMAGE_SIZE = (200, 200) ## tamaño al que se redimensionan las imágenes
    
    LEARNING_RATE = 0.001 ## tasa de aprendizaje
    HIDDEN_LAYER_SIZE = 60 ## tamaño de la capa oculta
    DEFAULT_EPOCHS = 300 ## número de épocas por defecto
    PREDICTION_THRESHOLD = 0.5 ## umbral de predicción: si la predicción > este valor = NEUMONIA, si <= este valor = NORMAL


    