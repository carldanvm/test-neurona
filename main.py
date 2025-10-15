from processImages import ImageProcessor

def showMenu():
    print("Menu")
    print("1. Procesar imágenes")
    print("2. Salir")
    option = int(input("Seleccione una opción: "))
    return option

def main():
    option = showMenu()
    while option != 2:
        if option == 1:
            processor = ImageProcessor()
            processor.process_all_original_images()
        option = showMenu()

if __name__ == "__main__":
    main()