import os
from PIL import Image
import numpy as np
import json

class Neurona:
    def __init__(self):
        self.training_data_dir = "imagenes-para-entrenar"
        self.test_data_dir = "imagenes-para-test"
        self.subdirectories = ["normal", "neumonia"]
        self.model_dir = "neurona"
        
        # Parámetros de la red neuronal
        self.weights1 = None  # Pesos de capa de entrada a capa oculta
        self.bias1 = None     # Bias de capa oculta
        self.weights2 = None  # Pesos de capa oculta a capa de salida
        self.bias2 = None     # Bias de capa de salida
        self.learning_rate = 0.001
        self.hidden_size = 60  # Número de neuronas en capa oculta
        self.test_accuracy = None  # Precisión en datos de test
        
        # Crear directorio para guardar modelo
        os.makedirs(self.model_dir, exist_ok=True)

    def process_training_data(self, balance=True):
        x = []  # Datos de imágenes
        y = []  # Etiquetas (0 = normal, 1 = neumonia)
        
        # Diccionario para almacenar imágenes por clase
        class_data = {0: [], 1: []}
        
        ## Convertir cada imagen en un array de intensidad de pixeles y normalizar a 0-1
        for idx, subdirectory in enumerate(self.subdirectories):
            input_subdir = os.path.join(self.training_data_dir, subdirectory)
            print(f"Procesando directorio: {subdirectory}")
            
            for filename in os.listdir(input_subdir):
                if any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp']):
                    input_path = os.path.join(input_subdir, filename)
                    try:
                        image = Image.open(input_path)
                        image_array = np.array(image)
                        image_array = image_array / 255.0
                        image_array = image_array.flatten()
                        
                        class_data[idx].append(image_array)
                        
                    except Exception as e:
                        print(f"Error procesando {input_path}: {e}")
        
        print(f"Imágenes cargadas - Normal: {len(class_data[0])}, Neumonia: {len(class_data[1])}")
        
        # Balancear dataset si es necesario
        if balance:
            min_samples = min(len(class_data[0]), len(class_data[1]))
            print(f"Balanceando dataset a {min_samples} imágenes por clase...")
            
            #Tomar muestra aleatoria de la clase mayoritaria
            np.random.seed(1)
            for class_idx in [0, 1]:
                if len(class_data[class_idx]) > min_samples:
                    indices = np.random.choice(len(class_data[class_idx]), min_samples, replace=False)
                    class_data[class_idx] = [class_data[class_idx][i] for i in indices]
        
        # Combinar todas las imágenes
        for class_idx in [0, 1]:
            for image_array in class_data[class_idx]:
                x.append(image_array)
                y.append(class_idx)
        
        # Shuffle de los datos
        x = np.array(x)
        y = np.array(y)
        
        shuffle_indices = np.random.permutation(len(x))
        x = x[shuffle_indices]
        y = y[shuffle_indices]
        
        print(f"Dataset final - Total: {len(x)}, Normal: {np.sum(y == 0)}, Neumonia: {np.sum(y == 1)}")
        
        return x, y
    
    def sigmoid(self, z):
        """Función de activación sigmoid"""
        # Evitar overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """Derivada de la función sigmoid"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def initialize_parameters(self, input_size):
        """Inicializar pesos y bias aleatoriamente"""
        # Capa 1: input -> hidden (entrada a capa oculta)
        self.weights1 = np.random.randn(input_size, self.hidden_size) * np.sqrt(2.0 / input_size)
        self.bias1 = np.zeros((1, self.hidden_size))
        
        # Capa 2: hidden -> output (capa oculta a salida)
        self.weights2 = np.random.randn(self.hidden_size, 1) * np.sqrt(2.0 / self.hidden_size)
        self.bias2 = 0.0
    
    def forward_propagation(self, X):
        """Propagación hacia adelante con capa oculta"""
        # Capa oculta
        z1 = np.dot(X, self.weights1) + self.bias1
        a1 = self.sigmoid(z1)  # Activación de neuronas ocultas
        
        # Capa de salida
        z2 = np.dot(a1, self.weights2) + self.bias2
        a2 = self.sigmoid(z2)  # Predicción final
        
        # Guardamos valores intermedios para backpropagation
        cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
        return cache
    
    def compute_cost(self, y_true, y_pred):
        """Calcular función de costo (cross-entropy)"""
        m = len(y_true)
        # Evitar log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cost = -1/m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
    
    def backward_propagation(self, X, y_true, cache):
        """Propagación hacia atrás con capa oculta"""
        m = len(y_true)
        
        # Extraer valores del cache
        a1 = cache['a1']
        a2 = cache['a2']
        
        # Reshape y_true para operaciones
        y_true = y_true.reshape(-1, 1)
        
        # Gradientes de capa de salida
        dz2 = a2 - y_true
        dw2 = 1/m * np.dot(a1.T, dz2)
        db2 = 1/m * np.sum(dz2)
        
        # Gradientes de capa oculta
        da1 = np.dot(dz2, self.weights2.T)
        dz1 = da1 * a1 * (1 - a1)  # Derivada de sigmoid
        dw1 = 1/m * np.dot(X.T, dz1)
        db1 = 1/m * np.sum(dz1, axis=0, keepdims=True)
        
        gradients = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}
        return gradients
    
    def update_parameters(self, gradients):
        """Actualizar pesos y bias de todas las capas"""
        self.weights1 -= self.learning_rate * gradients['dw1']
        self.bias1 -= self.learning_rate * gradients['db1']
        self.weights2 -= self.learning_rate * gradients['dw2']
        self.bias2 -= self.learning_rate * gradients['db2']
    
    def train(self, epochs=1000):
        """Entrenar la neurona"""
        print("Iniciando entrenamiento...")
        x, y = self.process_training_data(balance=True)
        
        # Inicializar parámetros
        self.initialize_parameters(x.shape[1])
        
        for epoch in range(epochs):
            # Forward propagation
            cache = self.forward_propagation(x)
            y_pred = cache['a2'].flatten()
            
            # Backward propagation
            gradients = self.backward_propagation(x, y, cache)
            
            # Actualizar parámetros
            self.update_parameters(gradients)
            
            # Mostrar progreso cada 100 épocas
            if epoch % 100 == 0:
                accuracy = self.calculate_accuracy(y, y_pred)
                print(f"Época {epoch}: Precisión = {accuracy:.2f}%")
        
        # Guardar modelo al finalizar
        self.save_model()
        print(f"\n¡Entrenamiento completado!")
        
        # Evaluar en datos de test
        print("\n" + "="*50)
        print("EVALUANDO CON DATOS DE TEST (no vistos)...")
        print("="*50)
        self.test_model()
    
    def calculate_accuracy(self, y_true, y_pred):
        """Calcular precisión del modelo"""
        predictions = (y_pred > 0.5).astype(int)
        accuracy = np.mean(predictions == y_true) * 100
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
        
        neuron_path = os.path.join(self.model_dir, 'neuron_parameters.json')
        with open(neuron_path, 'w') as f:
            json.dump(neuron_data, f, indent=2)
        
        print(f"Neurona guardada en: {neuron_path}")
    
    def load_model(self):
        """Cargar parámetros del modelo"""
        neuron_path = os.path.join(self.model_dir, 'neuron_parameters.json')
        
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
            # Procesar imagen igual que en entrenamiento
            image = Image.open(image_path)
            # IMPORTANTE: Redimensionar al mismo tamaño que el entrenamiento
            resized_image = image.resize((200, 200))  # Mismo tamaño que ImageProcessor
            processed_image = resized_image.convert("L")  # Convertir a escala de grises
            
            image_array = np.array(processed_image)
            image_array = image_array / 255.0
            image_array = image_array.flatten().reshape(1, -1)
            
            # Predecir
            cache = self.forward_propagation(image_array)
            prediction = cache['a2'][0][0]
            
            class_name = "NEUMONIA" if prediction > 0.5 else "NORMAL"
            
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
            
            for idx, subdirectory in enumerate(self.subdirectories):
                test_subdir = os.path.join(self.test_data_dir, subdirectory)
                
                if not os.path.exists(test_subdir):
                    print(f"Advertencia: No existe la carpeta {test_subdir}")
                    continue
                
                print(f"Cargando imágenes de test: {subdirectory}...")
                
                # Listar todas las imágenes
                all_files = [f for f in os.listdir(test_subdir) 
                           if any(f.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp'])]
                
                # Tomar muestra aleatoria (máximo max_samples)
                np.random.seed(1)  # Para reproducibilidad
                if len(all_files) > max_samples:
                    selected_files = np.random.choice(all_files, max_samples, replace=False)
                else:
                    selected_files = all_files
                
                print(f"  Procesando {len(selected_files)} imágenes de {subdirectory}...")
                
                for filename in selected_files:
                    test_path = os.path.join(test_subdir, filename)
                    try:
                        image = Image.open(test_path)
                        # IMPORTANTE: Procesar igual que en entrenamiento y predicción
                        resized_image = image.resize((200, 200))  # Mismo tamaño que ImageProcessor
                        processed_image = resized_image.convert("L")  # Convertir a escala de grises
                        
                        image_array = np.array(processed_image)
                        image_array = image_array / 255.0
                        image_array = image_array.flatten()
                        
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
            x_test = []
            y_test = []
            
            for class_idx in [0, 1]:
                for i in range(min_samples):
                    x_test.append(test_data[class_idx][i])
                    y_test.append(class_idx)
            
            x_test = np.array(x_test)
            y_test = np.array(y_test)
            
            # Hacer predicciones
            print(f"\nRealizando predicciones en {len(x_test)} imágenes de test...")
            cache = self.forward_propagation(x_test)
            y_pred = cache['a2'].flatten()
            
            # Calcular métricas
            predictions = (y_pred > 0.5).astype(int)
            
            # Precisión general
            accuracy = np.mean(predictions == y_test) * 100
            
            # Precisión por clase
            normal_mask = y_test == 0
            pneumonia_mask = y_test == 1
            
            normal_correct = np.sum(predictions[normal_mask] == 0)
            normal_total = np.sum(normal_mask)
            normal_accuracy = (normal_correct / normal_total * 100) if normal_total > 0 else 0
            
            pneumonia_correct = np.sum(predictions[pneumonia_mask] == 1)
            pneumonia_total = np.sum(pneumonia_mask)
            pneumonia_accuracy = (pneumonia_correct / pneumonia_total * 100) if pneumonia_total > 0 else 0
            
            # Guardar precisión de test
            self.test_accuracy = float(accuracy)
            
            # Mostrar resultados
            print("\n" + "="*50)
            print("RESULTADOS EN DATOS DE TEST")
            print("="*50)
            print(f"Total de imágenes evaluadas: {len(x_test)}")
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