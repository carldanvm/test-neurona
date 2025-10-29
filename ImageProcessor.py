import os
from PIL import Image
import numpy as np
from config import Config

class ImageProcessor:
    """
    Clase para redimensionar, convertir a escala de grises y normalizar las imágenes.
    """
    
    def __init__(self):
        """
        Inicializa el procesador de imágenes con las configuraciones del archivo config.py
        """
        self.input_dir = Config.RAW_TRAINING_IMAGES_DIR ## Directorio de las imágenes para entrenar originales
        self.output_dir = Config.TRAINING_IMAGES_DIR ## Directorio donde se guardaran las imágenes para entrenar procesadas
        self.subdirectories = Config.SUBDIRECTORIES ## Subdirectorios de las carpetas de entrenamiento originales y procesadas
        self.target_size = Config.IMAGE_SIZE ## Tamaño al que se redimensionan las imágenes
        self.supported_formats = Config.SUPPORTED_FORMATS ## Formatos de las imágenes soportados

    def resize_image(self, image_path, output_path):    
        """
        Redimensiona una imagen al tamaño especificado en config y la convierte a escala de grises.
        
        Args:
            image_path (str): Ruta de la imagen original
            output_path (str): Ruta donde se guardará la imagen procesada
        """
        try:
            # Abrir la imagen original
            image = Image.open(image_path)
            
            # Redimensionar la imagen al tamaño especificado en config
            resized_image = image.resize(self.target_size)
            
            # Convertir a escala de grises
            processed_image = resized_image.convert("L")
            
            # Guardar la imagen procesada en el directorio de salida
            processed_image.save(output_path)
        except Exception as e:
            print(f"Error procesando imagen {image_path}: {str(e)}")
        
    def process_all_original_images(self):
        """
        Procesa todas las imágenes originales del directorio de entrada.
        Redimensiona cada imagen y las convierte a escala de grises.
        Las imágenes procesadas se guardan en el directorio de salida.
        """
        print("Procesando todas las imágenes originales...")
        
        # Procesar cada subdirectorio
        for subdirectory in self.subdirectories:
            # Crear rutas completas de entrada y salida
            input_subdir = os.path.join(self.input_dir, subdirectory)
            output_subdir = os.path.join(self.output_dir, subdirectory)
            
            # Crear directorio de salida si no existe
            os.makedirs(output_subdir, exist_ok=True)
            
            # Procesar cada archivo en el subdirectorio
            for filename in os.listdir(input_subdir):
                # Verificar si el archivo tiene un formato soportado
                if any(filename.lower().endswith(ext) for ext in self.supported_formats):
                    # Crear rutas completas del archivo de entrada y salida
                    input_path = os.path.join(input_subdir, filename)
                    output_path = os.path.join(output_subdir, filename)
                    
                    # Procesar la imagen (redimensionar y convertir a escala de grises)
                    self.resize_image(input_path, output_path)
                    
        print("Todas las imágenes originales procesadas.")
    
    def process_image_for_prediction(self, image_path):
        """
        Procesa una imagen para ser usada en predicción o entrenamiento de la red neuronal.
        Aplica el mismo procesamiento que se usa durante el entrenamiento para mantener consistencia.
        
        Args:
            image_path (str): Ruta de la imagen a procesar
        
        Returns:
            numpy.ndarray: Array normalizado (valores entre 0-1) listo para la red neuronal.
                          None si hay error en el procesamiento.
        """
        try:
            # Abrir la imagen desde el archivo
            image = Image.open(image_path)
            
            # Redimensionar al mismo tamaño que se usa en entrenamiento
            resized_image = image.resize(self.target_size)
            
            # Convertir a escala de grises
            processed_image = resized_image.convert("L")
            
            # Convertir imagen PIL a array numpy
            image_array = np.array(processed_image)
            
            # Normalizar píxeles: de rango [0-255] a [0-1]
            # Esto es esencial para el entrenamiento de la red neuronal
            image_array = image_array / 255.0
            
            # Aplanar el array
            image_array = image_array.flatten()
            
            return image_array
            
        except Exception as e:
            print(f"Error procesando imagen {image_path}: {str(e)}")
            return None
            