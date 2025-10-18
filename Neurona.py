import os
from PIL import Image
import numpy as np
import json

class Neurona:
    def __init__(self):
        self.training_data_dir = "imagenes-para-entrenar"
        self.subdirectories = ["normal", "neumonia"]
        self.model_dir = "neurona"
        
        # Parámetros de la red neuronal
        self.weights = None
        self.bias = None
        self.learning_rate = 0.01
        
        # Crear directorio para guardar modelo
        os.makedirs(self.model_dir, exist_ok=True)

    def process_training_data(self):
        x = []  # Datos de imágenes
        y = []  # Etiquetas (0 = normal, 1 = neumonia)
        
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
                        
                        x.append(image_array)
                        y.append(idx)  # 0 para "normal", 1 para "neumonia"
                        
                    except Exception as e:
                        print(f"Error procesando {input_path}: {e}")
        
        print(f"Total imágenes procesadas: {len(x)}")
        print(f"Normal: {y.count(0)}, Neumonia: {y.count(1)}")
        
        return np.array(x), np.array(y)
    
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
        # Inicialización Xavier/Glorot
        self.weights = np.random.randn(input_size) * np.sqrt(2.0 / input_size)
        self.bias = 0.0
    
    def forward_propagation(self, X):
        """Propagación hacia adelante"""
        z = np.dot(X, self.weights) + self.bias
        a = self.sigmoid(z)
        return z, a
    
    def compute_cost(self, y_true, y_pred):
        """Calcular función de costo (cross-entropy)"""
        m = len(y_true)
        # Evitar log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cost = -1/m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
    
    def backward_propagation(self, X, y_true, y_pred):
        """Propagación hacia atrás"""
        m = len(y_true)
        dw = 1/m * np.dot(X.T, (y_pred - y_true))
        db = 1/m * np.sum(y_pred - y_true)
        return dw, db
    
    def update_parameters(self, dw, db):
        """Actualizar pesos y bias"""
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def train(self, epochs=1000):
        """Entrenar la neurona"""
        print("Iniciando entrenamiento...")
        x, y = self.process_training_data()
        
        # Inicializar parámetros
        self.initialize_parameters(x.shape[1])
        
        costs = []
        
        for epoch in range(epochs):
            # Forward propagation
            z, y_pred = self.forward_propagation(x)
            
            # Calcular costo
            cost = self.compute_cost(y, y_pred)
            costs.append(cost)
            
            # Backward propagation
            dw, db = self.backward_propagation(x, y, y_pred)
            
            # Actualizar parámetros
            self.update_parameters(dw, db)
            
            # Mostrar progreso cada 100 épocas
            if epoch % 100 == 0:
                accuracy = self.calculate_accuracy(y, y_pred)
                print(f"Época {epoch}: Costo = {cost:.4f}, Precisión = {accuracy:.2f}%")
        
        # Guardar modelo entrenado
        self.save_model()
        
        print(f"Entrenamiento completado. Costo final: {costs[-1]:.4f}")
        return costs
    
    def calculate_accuracy(self, y_true, y_pred):
        """Calcular precisión del modelo"""
        predictions = (y_pred > 0.5).astype(int)
        accuracy = np.mean(predictions == y_true) * 100
        return accuracy
    
    def save_model(self):
        """Guardar parámetros del modelo"""
        neuron_data = {
            'weights': self.weights.tolist(),
            'bias': float(self.bias),
            'learning_rate': self.learning_rate
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
            
            self.weights = np.array(neuron_data['weights'])
            self.bias = neuron_data['bias']
            self.learning_rate = neuron_data['learning_rate']
            
            print(f"Neurona cargada desde: {neuron_path}")
            return True
        else:
            print("No se encontró neurona guardada")
            return False
    
    def predict(self, image_path):
        """Predecir clase de una imagen"""
        if self.weights is None:
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
            _, prediction = self.forward_propagation(image_array)
            
            class_name = "neumonia" if prediction[0] > 0.5 else "normal"
            confidence = prediction[0] if prediction[0] > 0.5 else 1 - prediction[0]
            
            return {
                'class': class_name,
                'confidence': float(confidence),
                'probability': float(prediction[0])
            }
        
        except Exception as e:
            print(f"Error en predicción: {e}")
            return None


        