import os
from PIL import Image

class ImageProcessor:
    def __init__(self):
        self.input_dir = "imagenes-sin-procesar"
        self.output_dir = "imagenes-para-entrenar"
        self.subdirectories = ["neumonia", "normal"]
        self.target_size = (100, 100)

    def resize_image(self, image_path, output_path):
        try:
            image = Image.open(image_path)
            resized_image = image.resize(self.target_size)
            pgm_image = resized_image.convert("L")
            pgm_image.save(output_path)
        except Exception as e:
            print(f"Error procesando imagen {image_path}: {str(e)}")
        
    def process_all_original_images(self):
        print("Procesando todas las imágenes originales...")
        for subdirectory in self.subdirectories:
            input_subdir = os.path.join(self.input_dir, subdirectory)
            output_subdir = os.path.join(self.output_dir, subdirectory)
            
            # Crear directorio de salida si no existe
            os.makedirs(output_subdir, exist_ok=True)
            for filename in os.listdir(input_subdir):
                if any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp']):
                    input_path = os.path.join(input_subdir, filename)
                    output_path = os.path.join(output_subdir, filename)
                    self.resize_image(input_path, output_path)
        print("Todas las imágenes originales procesadas.")
            