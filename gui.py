import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import threading
import os
from Neurona import Neurona
from config import Config
from ImageProcessor import ImageProcessor

# Configuración global de CustomTkinter
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class NeuronaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Detección de Neumonía")
        self.root.geometry("1000x700")
        self.root.minsize(900, 600)
        
        # Configurar grid layout principal
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Variables
        self.current_image_path = None
        self.neurona = Neurona()
        self.current_ctk_image = None # Para mantener referencia a la imagen
        
        # Crear interfaz
        self.create_widgets()
        
        # Cargar info inicial
        self.update_neuron_info()
        
    def create_widgets(self):
        # --- Sidebar (Panel Izquierdo) ---
        self.sidebar_frame = ctk.CTkFrame(self.root, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1) # Espacio flexible abajo

        # Título en Sidebar
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Neurona AI", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Info de la Neurona
        self.info_frame = ctk.CTkFrame(self.sidebar_frame)
        self.info_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        self.info_label_title = ctk.CTkLabel(self.info_frame, text="Estado del Modelo", font=ctk.CTkFont(size=14, weight="bold"))
        self.info_label_title.pack(pady=(10, 5), padx=10)
        
        self.info_label_text = ctk.CTkLabel(self.info_frame, text="Cargando...", font=ctk.CTkFont(size=12), justify="left")
        self.info_label_text.pack(pady=(0, 10), padx=10)

        # Botones de Acción
        self.load_btn = ctk.CTkButton(self.sidebar_frame, text="Cargar Imagen", command=self.show_prediction_modal)
        self.load_btn.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        self.train_btn = ctk.CTkButton(self.sidebar_frame, text="Entrenar Neurona", command=self.train_neuron)
        self.train_btn.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        # Barra de progreso (oculta por defecto)
        self.progress = ctk.CTkProgressBar(self.sidebar_frame)
        self.progress.grid(row=6, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.progress.set(0)
        self.progress.grid_remove()

        # --- Área Principal (Panel Derecho) ---
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent") # Transparent para usar el fondo de la ventana
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Frame para la imagen
        self.image_frame = ctk.CTkFrame(self.main_frame, fg_color=("gray90", "gray16")) # Color de fondo sutil
        self.image_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        
        self.image_label = ctk.CTkLabel(self.image_frame, text="Carga una imagen para comenzar", font=ctk.CTkFont(size=16))
        self.image_label.grid(row=0, column=0)

        # Panel de Resultado y Acción
        self.action_frame = ctk.CTkFrame(self.main_frame, height=100)
        self.action_frame.grid(row=1, column=0, sticky="ew", padx=20)
        self.action_frame.grid_columnconfigure(0, weight=1)
        self.action_frame.grid_columnconfigure(1, weight=0)

        # Resultado Texto
        self.result_label = ctk.CTkLabel(self.action_frame, text="", font=ctk.CTkFont(size=20, weight="bold"))
        self.result_label.grid(row=0, column=0, padx=20, pady=20, sticky="w")
        
        # Botón Analizar
        self.predict_btn = ctk.CTkButton(self.action_frame, text="ANALIZAR IMAGEN", 
                                         font=ctk.CTkFont(size=15, weight="bold"),
                                         height=40,
                                         state="disabled",
                                         fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), # Estilo outline
                                         command=self.predict_image)
        self.predict_btn.grid(row=0, column=1, padx=20, pady=20)


    def update_neuron_info(self):
        try:
            import json
            neuron_path = os.path.join(Config.MODEL_DIR, 'neuron_parameters.json')
            
            if os.path.exists(neuron_path):
                with open(neuron_path, 'r') as f:
                    neuron_data = json.load(f)
                
                test_acc = neuron_data.get('test_accuracy', 0)
                text = f"Precisión: {test_acc:.1f}%\nEpochs: {Config.DEFAULT_EPOCHS}\nEstado: Listo"
                self.info_label_text.configure(text=text, text_color=("green", "#2CC985"), font=ctk.CTkFont(size=12, weight="bold"))
                self.train_btn.configure(text="Re-Entrenar Modelo")
                # Habilitar carga de imágenes si hay modelo
                self.load_btn.configure(state="normal")
            else:
                self.info_label_text.configure(text="Modelo no entrenado", text_color="gray", font=ctk.CTkFont(size=12, weight="bold"))
                self.train_btn.configure(text="Entrenar Modelo")
                # Deshabilitar carga de imágenes si no hay modelo
                self.load_btn.configure(state="disabled")
        except Exception:
            self.info_label_text.configure(text="Error leyendo modelo", text_color="red", font=ctk.CTkFont(size=12, weight="bold"))
            self.load_btn.configure(state="disabled")

    def load_image(self, filepath):
        # Verificar si el modelo está listo antes de cargar imagen
        if not os.path.exists(os.path.join(Config.MODEL_DIR, 'neuron_parameters.json')):
             messagebox.showwarning("Advertencia", "Debes entrenar la neurona antes de cargar imágenes.")
             return

        try:
            self.current_image_path = filepath
            # Cargar con PIL
            pil_image = Image.open(filepath)
            
            # Crear CTkImage (maneja escalado DPI)
            self.current_ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(400, 400))
            
            self.image_label.configure(image=self.current_ctk_image, text="")
            
            # Habilitar botón (Color Verde llamativo)
            self.predict_btn.configure(state="normal", fg_color="#2CC985", hover_color="#25A96F", text_color="white") 
            
            # Resetear resultado
            self.result_label.configure(text="Imagen cargada. Lista para análisis.", text_color=("gray50", "gray70"))
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{e}")

    def show_prediction_modal(self):
        # Modal simple con CTkToplevel
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Cargar Imagen")
        dialog.geometry("400x350")
        dialog.grab_set() # Modal
        
        ctk.CTkLabel(dialog, text="Selecciona fuente de imagen", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=20)
        
        def load_local():
            dialog.destroy()
            path = filedialog.askopenfilename(initialdir=Config.TEST_IMAGES_DIR ,filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif *.webp")])
            if path: self.load_image(path)
            
        ctk.CTkButton(dialog, text="Cargar desde Archivo", command=load_local).pack(pady=10, fill="x", padx=40)
        
        ctk.CTkLabel(dialog, text="O usar imagen de prueba:", font=ctk.CTkFont(size=14)).pack(pady=(20, 10))
        
        def load_test(type_):
             dialog.destroy()
             # Lógica para buscar imagen de test
             base_dir = os.path.join(Config.TEST_IMAGES_DIR, type_)
             if os.path.exists(base_dir):
                 path = filedialog.askopenfilename(initialdir=base_dir, title=f"Seleccionar imagen {type_}", 
                                                   filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif *.webp")])
                 if path: self.load_image(path)
             else:
                 messagebox.showwarning("Alerta", f"No se encontró el directorio {base_dir}")
        
        frame_btns = ctk.CTkFrame(dialog, fg_color="transparent")
        frame_btns.pack(pady=10)
        
        ctk.CTkButton(frame_btns, text="Neumonía", fg_color="#ef5350", hover_color="#d32f2f", command=lambda: load_test("neumonia")).pack(side="left", padx=5)
        ctk.CTkButton(frame_btns, text="Normal", fg_color="#66bb6a", hover_color="#43a047", command=lambda: load_test("normal")).pack(side="left", padx=5)

    def process_images(self):
        if not messagebox.askyesno("Confirmar", "¿Procesar todas las imágenes? Esto puede tardar."): return
        self.progress.grid()
        self.progress.start()
        self.disable_buttons()
        
        def _proc():
            try:
                processor = ImageProcessor()
                processor.process_all_original_images()
                self.root.after(0, lambda: messagebox.showinfo("Éxito", "Imágenes procesadas."))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, self.progress.stop)
                self.root.after(0, self.progress.grid_remove)
                self.root.after(0, self.enable_buttons)
        
        threading.Thread(target=_proc, daemon=True).start()

    def predict_image(self):
        if not self.current_image_path: return
        
        self.progress.grid()
        self.progress.start()
        self.disable_buttons()
        self.result_label.configure(text="Analizando...", text_color=("gray50", "gray70"))
        
        def _run():
            try:
                res = self.neurona.predict(self.current_image_path)
                if res:
                     prob = float(res['probability'])
                     if prob >= Config.PREDICTION_THRESHOLD:
                         txt, col = "NEUMONÍA", "#ef5350" # Red
                     else:
                         txt, col = "SIN NEUMONÍA", "#66bb6a" # Green
                     
                     self.root.after(0, lambda: self.result_label.configure(text=txt, text_color=col))
                else:
                     self.root.after(0, lambda: self.result_label.configure(text="Error en predicción", text_color="orange"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, self.progress.stop)
                self.root.after(0, self.progress.grid_remove)
                self.root.after(0, self.enable_buttons)
        
        threading.Thread(target=_run, daemon=True).start()

    def train_neuron(self):
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Entrenamiento")
        dialog.geometry("300x200")
        dialog.grab_set()
        
        ctk.CTkLabel(dialog, text="Configurar Entrenamiento", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
        ctk.CTkLabel(dialog, text="Número de épocas:").pack()
        
        entry = ctk.CTkEntry(dialog)
        entry.insert(0, str(Config.DEFAULT_EPOCHS))
        entry.pack(pady=10)
        entry.focus()
        
        def _start():
            try:
                epochs = int(entry.get())
                dialog.destroy()
                self.run_training(epochs)
            except ValueError:
                messagebox.showerror("Error", "Número inválido")
        
        ctk.CTkButton(dialog, text="Iniciar", command=_start).pack(pady=10)
        entry.bind('<Return>', lambda e: _start())

    def run_training(self, epochs):
        self.progress.grid()
        self.progress.start()
        self.disable_buttons()
        
        def _train():
            try:
                self.neurona.train(epochs)
                self.root.after(0, lambda: messagebox.showinfo("Info", f"Entrenamiento Finalizado.\nPrecisión: {self.neurona.test_accuracy:.1f}%"))
                self.root.after(0, self.update_neuron_info)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, self.progress.stop)
                self.root.after(0, self.progress.grid_remove)
                self.root.after(0, self.enable_buttons)

        threading.Thread(target=_train, daemon=True).start()

    def disable_buttons(self):
        self.train_btn.configure(state="disabled")
        self.predict_btn.configure(state="disabled")
        self.load_btn.configure(state="disabled")

    def enable_buttons(self):
        self.train_btn.configure(state="normal")
        self.predict_btn.configure(state="normal")
        self.load_btn.configure(state="normal")

def main():
    app = ctk.CTk()
    gui = NeuronaGUI(app)
    app.mainloop()
