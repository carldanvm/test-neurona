import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import os
from ImageProcessor import ImageProcessor
from Neurona import Neurona
from config import Config

class NeuronaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Detección de Neumonía")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.current_image_path = None
        self.neurona = Neurona()
        
        # Configurar estilo
        self.setup_styles()
        
        # Crear interfaz
        self.create_widgets()
        
        # Configurar drag & drop
        self.setup_drag_drop()
        
    def setup_styles(self):
        """Configurar estilos de ttk"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Estilo para botones principales
        style.configure('Primary.TButton',
                       background='#4CAF50',
                       foreground='white',
                       padding=10,
                       font=('Arial', 11, 'bold'))
        
        # Estilo para botón de predicción
        style.configure('Predict.TButton',
                       background='#2196F3',
                       foreground='white',
                       padding=10,
                       font=('Arial', 11, 'bold'))
        
    def create_widgets(self):
        """Crear todos los widgets de la interfaz"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Título
        title_label = tk.Label(main_frame, 
                              text="🫁 Sistema de Detección de Neumonía",
                              font=('Arial', 20, 'bold'),
                              bg='#f0f0f0',
                              fg='#333333')
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Frame de botones de procesamiento
        buttons_frame = ttk.LabelFrame(main_frame, text="Operaciones", padding="10")
        buttons_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        
        # Botón procesar imágenes
        self.process_btn = ttk.Button(buttons_frame,
                                     text="📁 Procesar Imágenes",
                                     command=self.process_images,
                                     style='Primary.TButton')
        self.process_btn.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Botón entrenar
        self.train_btn = ttk.Button(buttons_frame,
                                   text="🧠 Entrenar Neurona",
                                   command=self.train_neuron,
                                   style='Primary.TButton')
        self.train_btn.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Frame de predicción
        prediction_frame = ttk.LabelFrame(main_frame, text="Predicción", padding="10")
        prediction_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        prediction_frame.columnconfigure(0, weight=1)
        prediction_frame.rowconfigure(1, weight=1)
        
        # Área de drag & drop
        self.drop_frame = tk.Frame(prediction_frame, 
                                  bg='#e3f2fd',
                                  relief=tk.GROOVE,
                                  borderwidth=2)
        self.drop_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.drop_label = tk.Label(self.drop_frame,
                                  text="🖼️ Arrastra una imagen aquí\no haz clic para seleccionar",
                                  bg='#e3f2fd',
                                  fg='#1976d2',
                                  font=('Arial', 12),
                                  pady=30,
                                  cursor='hand2')
        self.drop_label.pack(fill=tk.BOTH, expand=True)
        self.drop_label.bind('<Button-1>', lambda e: self.select_image())
        
        # Frame para imagen y resultado
        result_container = ttk.Frame(prediction_frame)
        result_container.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        result_container.columnconfigure(0, weight=1)
        result_container.columnconfigure(1, weight=1)
        result_container.rowconfigure(0, weight=1)
        
        # Frame para vista previa de imagen
        image_frame = ttk.LabelFrame(result_container, text="Imagen", padding="10")
        image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        self.image_label = tk.Label(image_frame, text="Sin imagen", bg='white')
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Frame para resultado
        result_frame = ttk.LabelFrame(result_container, text="Resultado", padding="10")
        result_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        self.result_text = tk.Text(result_frame, 
                                  height=8,
                                  font=('Arial', 11),
                                  wrap=tk.WORD,
                                  state='disabled')
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Botón predecir
        self.predict_btn = ttk.Button(prediction_frame,
                                     text="🔍 Analizar Imagen",
                                     command=self.predict_image,
                                     style='Predict.TButton',
                                     state='disabled')
        self.predict_btn.grid(row=2, column=0, pady=(10, 0), sticky=(tk.W, tk.E))
        
        # Frame de log
        log_frame = ttk.LabelFrame(main_frame, text="Registro de Actividad", padding="10")
        log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Área de texto con scroll
        self.log_text = scrolledtext.ScrolledText(log_frame,
                                                 height=8,
                                                 font=('Consolas', 9),
                                                 wrap=tk.WORD,
                                                 state='disabled')
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Barra de progreso
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def setup_drag_drop(self):
        """Configurar drag & drop para el área de imagen"""
        # Intentar usar tkinterdnd2 si está disponible
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD
            # Si la ventana no es TkinterDnD, mostrar mensaje
            self.log("Drag & drop: usa el botón para seleccionar imagen")
        except ImportError:
            self.log("Drag & drop no disponible. Instala tkinterdnd2 para habilitar.")
        
        # Habilitar clic para seleccionar
        self.drop_frame.bind('<Button-1>', lambda e: self.select_image())
        
    def log(self, message):
        """Agregar mensaje al log"""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')
        
    def select_image(self):
        """Abrir diálogo para seleccionar imagen"""
        filetypes = [
            ('Imágenes', '*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif *.webp'),
            ('Todos los archivos', '*.*')
        ]
        filepath = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=filetypes
        )
        
        if filepath:
            self.load_image(filepath)
            
    def load_image(self, filepath):
        """Cargar y mostrar imagen"""
        try:
            self.current_image_path = filepath
            self.log(f"Imagen cargada: {os.path.basename(filepath)}")
            
            # Cargar y mostrar imagen
            img = Image.open(filepath)
            img.thumbnail((300, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.configure(image=photo, text='')
            self.image_label.image = photo
            
            # Habilitar botón de predicción
            self.predict_btn.configure(state='normal')
            
            # Limpiar resultado anterior
            self.result_text.configure(state='normal')
            self.result_text.delete(1.0, tk.END)
            self.result_text.configure(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{str(e)}")
            self.log(f"Error al cargar imagen: {str(e)}")
            
    def process_images(self):
        """Procesar todas las imágenes originales"""
        result = messagebox.askyesno(
            "Procesar Imágenes",
            "¿Desea procesar todas las imágenes originales?\n\n"
            "Esto puede tardar varios minutos."
        )
        
        if result:
            self.log("Iniciando procesamiento de imágenes...")
            self.progress.start()
            self.disable_buttons()
            
            def process():
                try:
                    processor = ImageProcessor()
                    processor.process_all_original_images()
                    self.root.after(0, lambda: self.log("✓ Imágenes procesadas exitosamente"))
                    self.root.after(0, lambda: messagebox.showinfo("Éxito", "Imágenes procesadas correctamente"))
                except Exception as e:
                    self.root.after(0, lambda: self.log(f"✗ Error: {str(e)}"))
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Error al procesar:\n{str(e)}"))
                finally:
                    self.root.after(0, self.progress.stop)
                    self.root.after(0, self.enable_buttons)
                    
            thread = threading.Thread(target=process, daemon=True)
            thread.start()
            
    def train_neuron(self):
        """Entrenar la neurona"""
        # Diálogo para épocas
        dialog = tk.Toplevel(self.root)
        dialog.title("Configurar Entrenamiento")
        dialog.geometry("350x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Centrar diálogo
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Número de épocas:", font=('Arial', 11)).pack(pady=(0, 10))
        
        epochs_var = tk.StringVar(value=str(Config.DEFAULT_EPOCHS))
        epochs_entry = ttk.Entry(frame, textvariable=epochs_var, font=('Arial', 11), width=15)
        epochs_entry.pack(pady=(0, 20))
        epochs_entry.focus()
        
        def start_training():
            try:
                epochs = int(epochs_var.get())
                if epochs <= 0:
                    messagebox.showwarning("Advertencia", "El número de épocas debe ser mayor a 0")
                    return
                dialog.destroy()
                self.run_training(epochs)
            except ValueError:
                messagebox.showwarning("Advertencia", "Por favor ingrese un número válido")
                
        ttk.Button(frame, text="Iniciar Entrenamiento", 
                  command=start_training,
                  style='Primary.TButton').pack()
        
        # Enter para confirmar
        epochs_entry.bind('<Return>', lambda e: start_training())
        
    def run_training(self, epochs):
        """Ejecutar entrenamiento en thread separado"""
        self.log(f"Iniciando entrenamiento con {epochs} épocas...")
        self.progress.start()
        self.disable_buttons()
        
        def train():
            try:
                neurona = Neurona()
                # Redirigir prints al log
                import sys
                from io import StringIO
                
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                try:
                    neurona.train(epochs=epochs)
                    output = sys.stdout.getvalue()
                finally:
                    sys.stdout = old_stdout
                
                # Mostrar output en log
                for line in output.split('\n'):
                    if line.strip():
                        self.root.after(0, lambda l=line: self.log(l))
                
                self.root.after(0, lambda: messagebox.showinfo("Éxito", 
                    f"Entrenamiento completado!\n\n"
                    f"Épocas: {epochs}\n"
                    f"Precisión en test: {neurona.test_accuracy:.2f}%" if neurona.test_accuracy else ""))
                    
            except Exception as e:
                self.root.after(0, lambda: self.log(f"✗ Error en entrenamiento: {str(e)}"))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Error durante entrenamiento:\n{str(e)}"))
            finally:
                self.root.after(0, self.progress.stop)
                self.root.after(0, self.enable_buttons)
                
        thread = threading.Thread(target=train, daemon=True)
        thread.start()
        
    def predict_image(self):
        """Hacer predicción sobre la imagen actual"""
        if not self.current_image_path:
            messagebox.showwarning("Advertencia", "Por favor selecciona una imagen primero")
            return
            
        self.log(f"Analizando: {os.path.basename(self.current_image_path)}")
        self.progress.start()
        self.disable_buttons()
        
        def predict():
            try:
                resultado = self.neurona.predict(self.current_image_path)
                
                if resultado:
                    # Mostrar resultado
                    clase = resultado['class']
                    probabilidad = resultado['probability']
                    
                    # Determinar color según resultado
                    if clase == "NEUMONIA":
                        color = '#f44336'  # Rojo
                        emoji = '⚠️'
                    else:
                        color = '#4CAF50'  # Verde
                        emoji = '✓'
                    
                    resultado_texto = (
                        f"{emoji} RESULTADO {emoji}\n\n"
                        f"Diagnóstico: {clase}\n\n"
                        f"Confianza: {probabilidad:.2%}\n\n"
                        f"{'━' * 25}\n"
                        f"{'Neumonía detectada' if clase == 'NEUMONIA' else 'Radiografía normal'}"
                    )
                    
                    self.root.after(0, lambda: self.update_result(resultado_texto, color))
                    self.root.after(0, lambda: self.log(f"✓ Predicción: {clase} ({probabilidad:.2%})"))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "No se pudo procesar la imagen"))
                    self.root.after(0, lambda: self.log("✗ Error al procesar la imagen"))
                    
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.log(f"✗ Error en predicción: {error_msg}"))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Error durante predicción:\n{error_msg}"))
            finally:
                self.root.after(0, self.progress.stop)
                self.root.after(0, self.enable_buttons)
                
        thread = threading.Thread(target=predict, daemon=True)
        thread.start()
        
    def update_result(self, text, color='#333333'):
        """Actualizar texto de resultado"""
        self.result_text.configure(state='normal')
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, text)
        self.result_text.tag_add("result", "1.0", "end")
        self.result_text.tag_config("result", foreground=color, font=('Arial', 11, 'bold'))
        self.result_text.configure(state='disabled')
        
    def disable_buttons(self):
        """Deshabilitar todos los botones durante operaciones"""
        self.process_btn.configure(state='disabled')
        self.train_btn.configure(state='disabled')
        self.predict_btn.configure(state='disabled')
        
    def enable_buttons(self):
        """Habilitar botones después de operaciones"""
        self.process_btn.configure(state='normal')
        self.train_btn.configure(state='normal')
        if self.current_image_path:
            self.predict_btn.configure(state='normal')

def main():
    """Función principal para iniciar la aplicación"""
    # Intentar usar TkinterDnD si está disponible
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except ImportError:
        root = tk.Tk()
    
    app = NeuronaGUI(root)
    
    # Mensaje de bienvenida
    app.log("=== Sistema de Detección de Neumonía ===")
    app.log("Bienvenido al sistema de análisis de radiografías")
    app.log("")
    
    root.mainloop()

if __name__ == "__main__":
    main()
