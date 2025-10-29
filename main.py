from ImageProcessor import ImageProcessor
from Neurona import Neurona
from config import Config

def showMenu():
    print("\n=== SISTEMA DE DETECCIÓN DE NEUMONÍA ===")
    print("1. Procesar imágenes")
    print("2. Entrenar neurona")
    print("3. Hacer predicción")
    print("4. Salir")
    option = int(input("Seleccione una opción: "))
    return option

def train_neuron():
    """Entrenar la neurona con las imágenes procesadas"""
    try:
        neurona = Neurona()
        epochs = int(input("Ingrese número de épocas (recomendado: 300): ") or Config.DEFAULT_EPOCHS)
        print(f"Entrenando neurona por {epochs} épocas...")
        neurona.train(epochs=epochs)
        print("¡Entrenamiento completado exitosamente!")
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")

def make_prediction():
    """Hacer predicción sobre una imagen"""
    try:
        neurona = Neurona()
        image_path = input("Ingrese la ruta de la imagen: ").strip()
        
        if not image_path:
            print("Ruta de imagen no válida")
            return
            
        print("Analizando imagen...")
        resultado = neurona.predict(image_path)
        
        if resultado:
            print(f"\n=== RESULTADO ===")
            print(f"Predicción: {resultado['class']}")
        else:
            print("No se pudo procesar la imagen")
            
    except Exception as e:
        print(f"Error durante la predicción: {e}")

def main():
    option = showMenu()
    while option != 4:
        if option == 1:
            print("Procesando imágenes...")
            processor = ImageProcessor()
            processor.process_all_original_images()
            print("¡Imágenes procesadas exitosamente!")
            
        elif option == 2:
            train_neuron()
            
        elif option == 3:
            make_prediction()
            
        else:
            print("Opción no válida. Intente de nuevo.")
            
        input("\nPresione Enter para continuar...")
        option = showMenu()

if __name__ == "__main__":
    main()