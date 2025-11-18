import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
        self.root.geometry("900x600")
        self.root.configure(bg='#F8FAFC')
        self.root.resizable(True, True)
        self.root.minsize(800, 500)
        try:
            # Maximizar ventana por defecto en Windows
            self.root.state('zoomed')
        except Exception:
            pass
        
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
        # Escoger tema a usar
        style.theme_use('clam')
        
        # Colores de fondo
        style.configure('TFrame', background='#F8FAFC')
        style.configure('TLabelframe', background='#F8FAFC')
        style.configure('TLabelframe.Label', background='#F8FAFC', foreground='#0F172A', font=('Segoe UI', 10, 'bold'))
        style.configure('TLabel', background='#F8FAFC', foreground='#0F172A', font=('Segoe UI', 10))

        style.configure('Card.TLabelframe', background='#FFFFFF', borderwidth=1, relief='solid')
        style.configure('Card.TLabelframe.Label', background='#FFFFFF', foreground='#0F172A', font=('Segoe UI', 10, 'bold'))

        # Estilo de los botones
        style.configure('Primary.TButton',
                        background='#10B981',
                        foreground='white',
                        padding=(12, 8),
                        font=('Segoe UI', 10, 'bold'),
                        borderwidth=0,
                        relief='flat')
        style.map('Primary.TButton',
                  background=[('active', '#059669'), ('pressed', '#047857'), ('disabled', '#34D399')],
                  foreground=[('active', '#FFFFFF'), ('pressed', '#FFFFFF'), ('disabled', '#FFFFFF')])

        style.configure('Predict.TButton',
                        background='#2563EB',
                        foreground='white',
                        padding=(12, 8),
                        font=('Segoe UI', 10, 'bold'),
                        borderwidth=0,
                        relief='flat')
        style.map('Predict.TButton',
                  background=[('active', '#1D4ED8'), ('pressed', '#1E3A8A'), ('disabled', '#60A5FA')],
                  foreground=[('active', '#FFFFFF'), ('pressed', '#FFFFFF'), ('disabled', '#FFFFFF')])

        # Barra de carga
        style.configure('Modern.Horizontal.TProgressbar',
                        troughcolor='#E5E7EB',
                        background='#2563EB',
                        bordercolor='#E5E7EB',
                        lightcolor='#2563EB',
                        darkcolor='#2563EB')
        
    def create_widgets(self):
        """Crear todos los widgets de la interfaz"""
        # Crear canvas y scrollbar para hacer la interfaz scrolleable
        canvas = tk.Canvas(self.root, bg='#F8FAFC', highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        
        # Frame principal dentro del canvas
        main_frame = ttk.Frame(canvas, padding="15")
        
        # Configurar scrollbar
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Posicionar canvas y scrollbar
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Crear ventana en el canvas
        canvas_frame = canvas.create_window((0, 0), window=main_frame, anchor="nw")
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Función para actualizar el scrollregion
        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Ajustar el ancho del frame al ancho del canvas
            canvas_width = canvas.winfo_width()
            canvas.itemconfig(canvas_frame, width=canvas_width)
        
        main_frame.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", configure_scroll_region)
        
        # Habilitar scroll con rueda del mouse
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Título
        title_label = tk.Label(main_frame, 
                              text="Sistema de Detección de Neumonía",
                              font=('Segoe UI', 16, 'bold'),
                              bg='#F8FAFC',
                              fg='#0F172A')
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # Frame de información de neurona
        neuron_info_frame = ttk.LabelFrame(main_frame, text="Estado de la Neurona", padding="8", style='Card.TLabelframe')
        neuron_info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        neuron_info_frame.columnconfigure(0, weight=1)
        
        self.neuron_info_text = tk.Text(neuron_info_frame,
                                        height=2,
                                        font=('Segoe UI', 9),
                                        wrap=tk.WORD,
                                        state='disabled',
                                        bg='#FFFFFF',
                                        bd=0,
                                        relief='flat',
                                        highlightthickness=0)
        self.neuron_info_text.pack()
        
        # Frame de botones de procesamiento
        buttons_frame = ttk.LabelFrame(main_frame, text="Operaciones", padding="8", style='Card.TLabelframe')
        buttons_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        buttons_frame.columnconfigure(0, weight=1)
        
        # Botón entrenar
        self.train_btn = ttk.Button(buttons_frame,
                                   text="Entrenar Neurona",
                                   command=self.train_neuron,
                                   style='Primary.TButton',
                                   cursor='hand2')
        self.train_btn.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Frame de predicción
        prediction_frame = ttk.LabelFrame(main_frame, text="Predicción", padding="8", style='Card.TLabelframe')
        prediction_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        prediction_frame.columnconfigure(0, weight=1)
        
        # Botón para realizar predicción (abre modal de selección)
        self.open_predict_modal_btn = ttk.Button(prediction_frame,
                                                text="Cargar Imagen para Predicción",
                                                style='Predict.TButton',
                                                command=self.show_prediction_modal,
                                                cursor='hand2')
        self.open_predict_modal_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        
        # Frame para imagen y resultado
        result_container = ttk.Frame(prediction_frame)
        result_container.grid(row=1, column=0, sticky=(tk.W, tk.E))
        result_container.columnconfigure(0, weight=1)
        result_container.columnconfigure(1, weight=1)
        
        # Frame para vista previa de imagen
        image_frame = ttk.LabelFrame(result_container, text="Imagen", padding="5", style='Card.TLabelframe')
        image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        # Canvas de tamaño reducido para mejor responsividad
        self.image_canvas = tk.Canvas(image_frame, width=350, height=350, bg='white', highlightthickness=0, bd=0, relief='flat')
        self.image_canvas.pack()
        # Texto por defecto cuando no hay imagen
        self.image_canvas.create_text(175, 175, text="Sin imagen", fill="#999999", font=('Segoe UI', 11))
        
        # Frame para resultado
        result_frame = ttk.LabelFrame(result_container, text="Resultado", padding="5", style='Card.TLabelframe')
        result_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        
        self.result_text = tk.Text(result_frame, 
                                  height=10,
                                  width=25,
                                  font=('Segoe UI', 10),
                                  wrap=tk.WORD,
                                  state='disabled',
                                  bg='#FFFFFF',
                                  bd=0,
                                  relief='flat',
                                  highlightthickness=0)
        self.result_text.pack()
        
        # Botón predecir
        self.predict_btn = ttk.Button(prediction_frame,
                                     text="🔍 Analizar Imagen",
                                     command=self.predict_image,
                                     style='Predict.TButton',
                                     cursor='hand2',
                                     state='disabled')
        self.predict_btn.grid(row=2, column=0, pady=(8, 0), sticky=(tk.W, tk.E))
        
        # Barra de progreso
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', style='Modern.Horizontal.TProgressbar')
        self.progress.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(8, 10))
        
    def setup_drag_drop(self):
        """Inicialización posterior a widgets (sin drag & drop)"""
        # Cargar y mostrar información de neurona existente
        self.update_neuron_info()

    def show_prediction_modal(self):
        """Modal para elegir la fuente de la imagen a predecir"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Realizar Predicción")
        dialog.geometry("520x240")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Sección: Cargar imagen nueva
        ttk.Label(frame, text="Cargar imagen", font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        
        def choose_new_image():
            base_dir = os.path.join(Config.TEST_IMAGES_DIR)
            initial_dir = base_dir if os.path.isdir(base_dir) else os.getcwd()
            filetypes = [
                ('Imágenes', '*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif *.webp'),
                ('Todos los archivos', '*.*')
            ]
            path = filedialog.askopenfilename(title="Seleccionar imagen", initialdir=initial_dir, filetypes=filetypes)
            if path:
                dialog.destroy()
                self.load_image(path)
        ttk.Button(frame, text="Seleccionar una imagen", command=choose_new_image, style='Predict.TButton', cursor='hand2').pack(anchor='w', pady=(0, 12))
        
        # Separador
        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=6)
        
        # Sección: Cargar de imágenes de prueba
        ttk.Label(frame, text="Utilizar una imagen de prueba (resultado real conocido)", font=('Segoe UI', 11, 'bold')).pack(anchor='w', pady=(6, 4))
        ttk.Label(frame, text="Elige la clase de la imagen de prueba:", font=('Segoe UI', 10)).pack(anchor='w')
        
        btns = ttk.Frame(frame)
        btns.pack(fill=tk.X, pady=(6, 0))
        
        def choose_from_test(subdir):
            base_dir = os.path.join(Config.TEST_IMAGES_DIR, subdir)
            initial_dir = base_dir if os.path.isdir(base_dir) else Config.TEST_IMAGES_DIR
            filetypes = [
                ('Imágenes', '*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif *.webp'),
                ('Todos los archivos', '*.*')
            ]
            path = filedialog.askopenfilename(title="Seleccionar imagen de prueba", initialdir=initial_dir, filetypes=filetypes)
            if path:
                dialog.destroy()
                self.load_image(path)
        
        ttk.Button(btns, text="Neumonía", style='Primary.TButton', cursor='hand2', command=lambda: choose_from_test('neumonia')).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btns, text="Sin neumonía", style='Primary.TButton', cursor='hand2', command=lambda: choose_from_test('normal')).pack(side=tk.LEFT)
        
    def update_neuron_info(self):
        """Actualizar información de la neurona entrenada"""
        try:
            # Intentar cargar información de la neurona
            import os
            import json
            neuron_path = os.path.join(Config.MODEL_DIR, 'neuron_parameters.json')
            
            self.neuron_info_text.configure(state='normal')
            self.neuron_info_text.delete(1.0, tk.END)
            
            if os.path.exists(neuron_path):
                with open(neuron_path, 'r') as f:
                    neuron_data = json.load(f)
                
                learning_rate = neuron_data.get('learning_rate', 'N/A')
                hidden_size = neuron_data.get('hidden_size', 'N/A')
                test_accuracy = neuron_data.get('test_accuracy', None)
                
                info_text = (
                    f"✓ Neurona entrenada encontrada\n"
                    f"Learning Rate usado: {learning_rate} \n"
                    f"Capa Oculta: {hidden_size} neuronas \n"
                    f"Precisión en Test: {test_accuracy:.2f}%" if test_accuracy else f"Precisión en Test: N/A"
                )
                
                self.neuron_info_text.insert(1.0, info_text)
                self.neuron_info_text.tag_add("success", "1.0", "end")
                self.neuron_info_text.tag_config("success", foreground='#4CAF50', font=('Segoe UI', 10, 'bold'))
                
                # Cambiar texto del botón
                self.train_btn.configure(text="Entrenar Otra Vez")
            else:
                info_text = "⚠ No se encontró neurona entrenada. Por favor, entrena una neurona primero."
                self.neuron_info_text.insert(1.0, info_text)
                self.neuron_info_text.tag_add("warning", "1.0", "end")
                self.neuron_info_text.tag_config("warning", foreground='#FF9800', font=('Segoe UI', 10, 'bold'))
                
                # Texto original del botón
                self.train_btn.configure(text="Entrenar Neurona")
                
            self.neuron_info_text.configure(state='disabled')
            
        except Exception as e:
            print(f"Error al cargar información de neurona: {e}")
        
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
            
            # Cargar imagen y ajustarla para caber en el canvas (350x350) manteniendo proporciones
            img = Image.open(filepath)
            img = img.convert('RGB')
            img.thumbnail((350, 350), Image.Resampling.LANCZOS)
            self._img_photo = ImageTk.PhotoImage(img)
            
            # Limpiar canvas y dibujar imagen centrada
            self.image_canvas.delete("all")
            self.image_canvas.create_image(175, 175, image=self._img_photo)
            
            # Habilitar botón de predicción
            self.predict_btn.configure(state='normal')
            
            # Limpiar resultado anterior
            self.result_text.configure(state='normal')
            self.result_text.delete(1.0, tk.END)
            self.result_text.configure(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{str(e)}")
            
    def process_images(self):
        """Procesar todas las imágenes originales"""
        result = messagebox.askyesno(
            "Procesar Imágenes",
            "¿Desea procesar todas las imágenes originales?\n\n"
            "Esto puede tardar varios minutos."
        )
        
        if result:
            self.progress.start()
            self.disable_buttons()
            
            def process():
                try:
                    processor = ImageProcessor()
                    processor.process_all_original_images()
                    self.root.after(0, lambda: messagebox.showinfo("Éxito", "Imágenes procesadas correctamente"))
                except Exception as e:
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
        dialog.geometry("480x180")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Centrar diálogo
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Número de épocas (mejor resultado: 300):", font=('Segoe UI', 11)).pack(pady=(0, 10))
        
        epochs_var = tk.StringVar(value=str(Config.DEFAULT_EPOCHS))
        epochs_entry = ttk.Entry(frame, textvariable=epochs_var, font=('Segoe UI', 11), width=15)
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
                  style='Primary.TButton',
                  cursor='hand2').pack(fill=tk.X)
        
        # Enter para confirmar
        epochs_entry.bind('<Return>', lambda e: start_training())
        
    def run_training(self, epochs):
        """Ejecutar entrenamiento en thread separado"""
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
                finally:
                    sys.stdout = old_stdout
                
                self.root.after(0, lambda: messagebox.showinfo("Éxito", 
                    f"Entrenamiento completado!\n\n"
                    f"Épocas: {epochs}\n"
                    f"Precisión en test: {neurona.test_accuracy:.2f}%" if neurona.test_accuracy else ""))
                self.root.after(0, self.update_neuron_info)
                    
            except Exception as e:
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
            
        self.progress.start()
        self.disable_buttons()
        
        def predict():
            try:
                resultado = self.neurona.predict(self.current_image_path)
                
                if resultado:
                    # Mostrar solo probabilidad y color según umbrales
                    prob = float(resultado['probability'])
                    prob_pct = prob * 100
                    if prob_pct < 50:
                        color = '#4CAF50'  # Verde
                    elif prob_pct < 70:
                        color = '#FF9800'  # Naranja
                    else:
                        color = '#f44336'  # Rojo
                    resultado_texto = f"Probabilidad de neumonía: {prob_pct:.2f}%"
                    self.root.after(0, lambda: self.update_result(resultado_texto, color))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "No se pudo procesar la imagen"))
                    
            except Exception as e:
                error_msg = str(e)
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
        self.result_text.tag_config("result", foreground=color, font=('Segoe UI', 11, 'bold'))
        self.result_text.configure(state='disabled')
        
    def disable_buttons(self):
        """Deshabilitar todos los botones durante operaciones"""
        self.train_btn.configure(state='disabled')
        self.predict_btn.configure(state='disabled')
        
    def enable_buttons(self):
        """Habilitar botones después de operaciones"""
        self.train_btn.configure(state='normal')
        if self.current_image_path:
            self.predict_btn.configure(state='normal')

def main():
    """Función principal para iniciar la aplicación"""
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except ImportError:
        root = tk.Tk()
    
    app = NeuronaGUI(root)
    root.mainloop()
