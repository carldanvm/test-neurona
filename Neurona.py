import os
import numpy as np
import json
from ImageProcessor import ImageProcessor
from config import Config

class Neurona:
    def __init__(self):
        self.training_data_dir = Config.TRAINING_IMAGES_DIR ## Directorio de las imágenes para entrenar procesadas
        self.test_data_dir = Config.TEST_IMAGES_DIR ## Directorio de las imágenes para testing
        self.subdirectories = Config.SUBDIRECTORIES ## Subdirectorios de las carpetas de entrenamiento y testing
        self.model_dir = Config.MODEL_DIR ## Directorio donde se guardara el modelo
        
        # Parámetros de la red neuronal
        self.weights1 = None ## Pesos de capa de entrada a capa oculta
        self.bias1 = None ## Bias de capa oculta
        self.weights2 = None ## Pesos de capa oculta a capa de salida
        self.bias2 = None ## Bias de capa de salida
        self.learning_rate = Config.LEARNING_RATE ## Tasa de aprendizaje
        self.hidden_size = Config.HIDDEN_LAYER_SIZE ## Número de neuronas en capa oculta
        self.test_accuracy = None ## Precisión en datos de test
        self.prediction_threshold = Config.PREDICTION_THRESHOLD ## Umbral de predicción
        
        # Instancia de ImageProcessor para procesamiento de imágenes
        self.image_processor = ImageProcessor()
        self.supported_formats = Config.SUPPORTED_FORMATS ## Formatos de las imágenes soportados
        
        # Crear directorio para guardar modelo si no existe
        os.makedirs(self.model_dir, exist_ok=True)

    def process_training_data(self, balance=True):
        images_data = [] ## Datos de imágenes
        labels_data = [] ## Etiquetas (0 = normal, 1 = neumonia)
        
        ## Diccionario para almacenar imágenes por clase
        class_data = {0: [], 1: []}
        
        ## Convertir cada imagen en un array de intensidad de pixeles y normalizar a 0-1
        for idx, subdirectory in enumerate(self.subdirectories): ## idx: 0=normal, 1=neumonia
            input_subdir = os.path.join(self.training_data_dir, subdirectory)
            print(f"Procesando directorio: {subdirectory}")
            
            for filename in os.listdir(input_subdir):
                if any(filename.lower().endswith(ext) for ext in self.supported_formats): ## Filtrar solo imágenes válidas
                    input_path = os.path.join(input_subdir, filename)
                    try:
                        # Procesar imagen usando ImageProcessor centralizado
                        image_array = self.image_processor.process_image_for_prediction(input_path)
                        if image_array is not None:
                            class_data[idx].append(image_array) ## Agregar a la clase correspondiente
                        
                    except Exception as e:
                        print(f"Error procesando {input_path}: {e}")
        
        print(f"Imágenes cargadas - Normal: {len(class_data[0])}, Neumonia: {len(class_data[1])}")
        
        # Balancear dataset si es necesario
        if balance:
            min_samples = min(len(class_data[0]), len(class_data[1])) ## Encontrar la clase con menos imágenes
            print(f"Balanceando dataset a {min_samples} imágenes por clase...")
            
            #Tomar muestra aleatoria de la clase mayoritaria
            np.random.seed(1) ## Semilla para reproducibilidad
            for class_idx in [0, 1]:
                if len(class_data[class_idx]) > min_samples: ## Si esta clase tiene más imágenes que el mínimo
                    # Generar índices aleatorios para seleccionar 'min_samples' imágenes
                    indices = np.random.choice(len(class_data[class_idx]), min_samples, replace=False)
                    # Filtrar la lista manteniendo solo las imágenes en los índices seleccionados
                    class_data[class_idx] = [class_data[class_idx][i] for i in indices]
        
        # Combinar todas las imágenes
        for class_idx in [0, 1]:
            for image_array in class_data[class_idx]:
                images_data.append(image_array) ## Agregar imagen
                labels_data.append(class_idx) ## Agregar etiqueta correspondiente
        
        # Shuffle de los datos
        images_data = np.array(images_data)
        labels_data = np.array(labels_data)
        
        shuffle_indices = np.random.permutation(len(images_data)) ## Generar índices aleatorios
        images_data = images_data[shuffle_indices] ## Mezclar imágenes
        labels_data = labels_data[shuffle_indices] ## Mezclar etiquetas en el mismo orden
        
        print(f"Dataset final - Total: {len(images_data)}, Normal: {np.sum(labels_data == 0)}, Neumonia: {np.sum(labels_data == 1)}")
        
        return images_data, labels_data
    
    def sigmoid(self, linear_input):
        """Función de activación sigmoid"""
        # Evitar overflow limitando la entrada lineal a un rango seguro
        linear_input = np.clip(linear_input, -500, 500)
        # Calcular sigmoid: 1 / (1 + e^(-entrada_lineal))
        return 1 / (1 + np.exp(-linear_input))
    
    def sigmoid_derivative(self, linear_input):
        """Derivada de la función sigmoid"""
        # Calcular sigmoid primero
        sigmoid_value = self.sigmoid(linear_input)
        # La derivada de sigmoid es: sigmoid(entrada) * (1 - sigmoid(entrada))
        return sigmoid_value * (1 - sigmoid_value)
    
    def initialize_parameters(self, input_size):
        """Inicializar pesos y bias aleatoriamente"""
        # Capa 1: input -> hidden (entrada a capa oculta)
        # Inicialización, con metodo Xavier/He
        self.weights1 = np.random.randn(input_size, self.hidden_size) * np.sqrt(2.0 / input_size)
        # Bias de capa oculta inicializado en ceros
        self.bias1 = np.zeros((1, self.hidden_size))
        
        # Capa 2: hidden -> output (capa oculta a salida)
        # Pesos de salida también con inicialización Xavier/He
        self.weights2 = np.random.randn(self.hidden_size, 1) * np.sqrt(2.0 / self.hidden_size)
        # Bias de salida inicializado en cero
        self.bias2 = 0.0
    
    def forward_propagation(self, input_images):
        """Propagación hacia adelante con capa oculta"""
        # Capa oculta
        # Calcular entrada lineal: imagenes * pesos_capa1 + bias_capa1
        hidden_linear = np.dot(input_images, self.weights1) + self.bias1
        # Aplicar función de activación sigmoid
        hidden_activation = self.sigmoid(hidden_linear)  # Activación de neuronas ocultas
        
        # Capa de salida
        # Calcular entrada lineal de salida: activacion_oculta * pesos_salida + bias_salida
        output_linear = np.dot(hidden_activation, self.weights2) + self.bias2
        # Aplicar sigmoid para obtener probabilidad (0-1)
        output_activation = self.sigmoid(output_linear)  # Predicción final
        
        # Guardamos valores para backpropagation
        cache = {
            'hidden_linear': hidden_linear, 
            'hidden_activation': hidden_activation, 
            'output_linear': output_linear, 
            'output_activation': output_activation
        }
        return cache
    
    def backward_propagation(self, input_images, true_labels, cache):
        """Propagación hacia atrás con capa oculta"""
        num_samples = len(true_labels)
        
        # Extraer valores del cache
        hidden_activation = cache['hidden_activation']
        output_activation = cache['output_activation']
        
        # Reshape true_labels para operaciones
        true_labels = true_labels.reshape(-1, 1)
        
        # Gradientes de capa de salida
        output_error = output_activation - true_labels
        weights2_gradient = 1/num_samples * np.dot(hidden_activation.T, output_error)
        bias2_gradient = 1/num_samples * np.sum(output_error)
        
        # Gradientes de capa oculta
        hidden_error_signal = np.dot(output_error, self.weights2.T)
        hidden_error = hidden_error_signal * hidden_activation * (1 - hidden_activation)  # Derivada de sigmoid
        weights1_gradient = 1/num_samples * np.dot(input_images.T, hidden_error)
        bias1_gradient = 1/num_samples * np.sum(hidden_error, axis=0, keepdims=True)
        
        gradients = {
            'weights1_gradient': weights1_gradient, 
            'bias1_gradient': bias1_gradient, 
            'weights2_gradient': weights2_gradient, 
            'bias2_gradient': bias2_gradient
        }
        return gradients
    
    def update_parameters(self, gradients):
        """Actualizar pesos y bias de todas las capas"""
        self.weights1 -= self.learning_rate * gradients['weights1_gradient']
        self.bias1 -= self.learning_rate * gradients['bias1_gradient']
        self.weights2 -= self.learning_rate * gradients['weights2_gradient']
        self.bias2 -= self.learning_rate * gradients['bias2_gradient']
    
    def train(self, epochs=Config.DEFAULT_EPOCHS):
        """Entrenar la neurona"""
        print("Iniciando entrenamiento...")
        training_images, training_labels = self.process_training_data(balance=True)
        
        # Inicializar parámetros
        self.initialize_parameters(training_images.shape[1])
        
        # Entrenar la neurona por el numero de épocas
        for epoch in range(epochs):
            # Forward propagation
            cache = self.forward_propagation(training_images)
            predictions = cache['output_activation'].flatten()
            
            # Backward propagation
            gradients = self.backward_propagation(training_images, training_labels, cache)
            
            # Actualizar parámetros
            self.update_parameters(gradients)
            
            # Mostrar progreso cada 100 épocas
            if epoch % 100 == 0:
                accuracy = self.calculate_accuracy(training_labels, predictions)
                print(f"Época {epoch}: Precisión = {accuracy:.2f}%")
        
        # Guardar modelo al finalizar
        self.save_model()
        print(f"\n¡Entrenamiento completado!")
        
        # Evaluar en datos de test
        print("\n" + "="*50)
        print("EVALUANDO CON DATOS DE TEST (no vistos)...")
        print("="*50)
        self.test_model()
    
    def calculate_accuracy(self, true_labels, predicted_probabilities):
        """Calcular precisión del modelo"""
        # Convertir predicciones a 0-1
        binary_predictions = (predicted_probabilities > self.prediction_threshold).astype(int)
        # Calcular precisión
        accuracy = np.mean(binary_predictions == true_labels) * 100
        return accuracy
    
    
    def save_model(self):
        """Guardar parámetros del modelo"""
        neuron_data = {
            'weights1': self.weights1.tolist(),
            'bias1': self.bias1.tolist(),
            'weights2': self.weights2.tolist(),
            'bias2': float(self.bias2),
            'learning_rate': self.learning_rate,
            'hidden_size': self.hidden_size,
            'test_accuracy': self.test_accuracy
        }
        
        # Guardar modelo en JSON
        neuron_path = os.path.join(self.model_dir, 'neuron_parameters.json')
        with open(neuron_path, 'w') as f:
            json.dump(neuron_data, f, indent=2)
        
        print(f"Neurona guardada en: {neuron_path}")
    
    def load_model(self):
        """Cargar parámetros del modelo"""
        neuron_path = os.path.join(self.model_dir, 'neuron_parameters.json')
        
        # Cargar modelo desde JSON
        if os.path.exists(neuron_path):
            with open(neuron_path, 'r') as f:
                neuron_data = json.load(f)
            
            self.weights1 = np.array(neuron_data['weights1'])
            self.bias1 = np.array(neuron_data['bias1'])
            self.weights2 = np.array(neuron_data['weights2'])
            self.bias2 = neuron_data['bias2']
            self.learning_rate = neuron_data['learning_rate']
            self.hidden_size = neuron_data['hidden_size']
            self.test_accuracy = neuron_data.get('test_accuracy', None)
            
            print(f"Neurona cargada desde: {neuron_path}")
            return True
        else:
            print("No se encontró neurona guardada")
            return False
    
    def predict(self, image_path):
        """Predecir clase de una imagen"""
        if self.weights1 is None:
            if not self.load_model():
                raise ValueError("No hay modelo entrenado. Entrena primero o carga un modelo.")
        
        try:
            # Procesar imagen usando ImageProcessor centralizado
            image_array = self.image_processor.process_image_for_prediction(image_path)
            if image_array is None:
                return None
            image_array = image_array.reshape(1, -1)
            
            # Predecir
            cache = self.forward_propagation(image_array)
            prediction = cache['output_activation'][0][0]
            
            # Interpretar la prediccion, mayor a threshold es neumonia, menor es normal
            class_name = "NEUMONIA" if prediction > self.prediction_threshold else "NORMAL"
            
            return {
                'class': class_name,
                'probability': float(prediction)
            }
        
        except Exception as e:
            print(f"Error en predicción: {e}")
            return None
    
    def test_model(self, max_samples=1000):
        """Evaluar el modelo con datos de test no vistos"""
        try:
            # Cargar imágenes de test
            test_data = {0: [], 1: []}  # 0: normal, 1: neumonia
            test_labels = {0: [], 1: []}
            
            # Recorrer subdirectorios
            for idx, subdirectory in enumerate(self.subdirectories):
                # Directorio de imagenes de test
                test_subdir = os.path.join(self.test_data_dir, subdirectory)
                
                if not os.path.exists(test_subdir):
                    print(f"Advertencia: No existe la carpeta {test_subdir}")
                    continue
                
                print(f"Cargando imágenes de test: {subdirectory}...")
                
                # Listar todas las imágenes
                all_files = [f for f in os.listdir(test_subdir) 
                           if any(f.lower().endswith(ext) for ext in self.supported_formats)]
                
                # Tomar muestra aleatoria
                np.random.seed(1)  # Para reproducibilidad
                
                # Si hay mas archivos que max_samples, tomar una muestra aleatoria
                if len(all_files) > max_samples:
                    selected_files = np.random.choice(all_files, max_samples, replace=False)
                # Si hay menos archivos que max_samples, tomar todos los archivos
                else:
                    selected_files = all_files
                
                print(f"  Procesando {len(selected_files)} imágenes de {subdirectory}...")
                
                # Procesar cada imagen
                for filename in selected_files:
                    test_path = os.path.join(test_subdir, filename)
                    try:
                        # Procesar imagen usando ImageProcessor centralizado
                        image_array = self.image_processor.process_image_for_prediction(test_path)
                        if image_array is not None:
                            # Añadir a los datos de test
                            test_data[idx].append(image_array)
                            test_labels[idx].append(idx)
                    except Exception as e:
                        print(f"  Error procesando {test_path}: {e}")
            
            # Verificar que hay datos de test
            if len(test_data[0]) == 0 or len(test_data[1]) == 0:
                print("No se encontraron suficientes imágenes de test")
                return
            
            # Balancear: tomar la cantidad mínima de ambas clases
            min_samples = min(len(test_data[0]), len(test_data[1]))
            print(f"\nUsando {min_samples} imágenes de cada clase para testing")
            
            # Combinar datos
            test_images = []
            test_labels = []
            
            for class_idx in [0, 1]:
                for i in range(min_samples):
                    test_images.append(test_data[class_idx][i])
                    test_labels.append(class_idx)
            
            test_images = np.array(test_images)
            test_labels = np.array(test_labels)
            
            # Hacer predicciones
            print(f"\nRealizando predicciones en {len(test_images)} imágenes de test...")
            cache = self.forward_propagation(test_images)
            predicted_probabilities = cache['output_activation'].flatten()
            
            # Calcular métricas
            binary_predictions = (predicted_probabilities > self.prediction_threshold).astype(int)
            
            # Precisión general
            accuracy = np.mean(binary_predictions == test_labels) * 100
            
            # Precisión por clase
            normal_mask = test_labels == 0
            pneumonia_mask = test_labels == 1
            
            normal_correct = np.sum(binary_predictions[normal_mask] == 0)
            normal_total = np.sum(normal_mask)
            normal_accuracy = (normal_correct / normal_total * 100) if normal_total > 0 else 0
            
            pneumonia_correct = np.sum(binary_predictions[pneumonia_mask] == 1)
            pneumonia_total = np.sum(pneumonia_mask)
            pneumonia_accuracy = (pneumonia_correct / pneumonia_total * 100) if pneumonia_total > 0 else 0
            
            # Guardar precisión de test
            self.test_accuracy = float(accuracy)
            
            # Mostrar resultados
            print("\n" + "="*50)
            print("RESULTADOS EN DATOS DE TEST")
            print("="*50)
            print(f"Total de imágenes evaluadas: {len(test_images)}")
            print(f"  - Imágenes normales: {normal_total}")
            print(f"  - Imágenes con neumonía: {pneumonia_total}")
            print("\n" + "-"*50)
            print(f"PRECISIÓN REAL EN TEST: {accuracy:.2f}%")
            print("-"*50)
            print(f"Precisión en NORMALES: {normal_accuracy:.2f}% ({normal_correct}/{normal_total})")
            print(f"Precisión en NEUMONÍA: {pneumonia_accuracy:.2f}% ({pneumonia_correct}/{pneumonia_total})")
            print("="*50)
            
            # Guardar modelo con la precisión de test
            self.save_model()
            print(f"\nModelo guardado con precisión de test: {accuracy:.2f}%")
            
        except Exception as e:
            print(f"Error durante testing: {e}")
            import traceback
            traceback.print_exc()