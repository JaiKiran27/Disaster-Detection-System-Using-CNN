import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from keras.models import load_model
import os
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
import tkinter as tk
from tkinter import ttk
import threading
import time

class HomeScreen:
    def __init__(self, root):
        self.root = root
        self.root.title("Home Screen")
        self.root.state('zoomed')

        self.current_screen = None
        self.intro_screen = None

        self.show_intro_screen()

    def show_intro_screen(self):
        self.current_screen = self.intro_screen = tk.Frame(self.root)
        self.intro_screen.pack(fill="both", expand=True)

        image_path = "C:\\Users\\Asus\\OneDrive\\Desktop\\6th Sem Project\\Final Dependencies\\Home.jpg"
        image = Image.open(image_path)
        image = image.resize((1500, 820))
        image = ImageTk.PhotoImage(image)

        image_label = tk.Label(self.intro_screen, image=image)
        image_label.image = image
        image_label.place(x=0, y=0, relwidth=1, relheight=1)

        button_frame = tk.Frame(self.intro_screen, bg="#1f497d")
        button_frame.place(relx=0.73, rely=0.85, anchor=tk.CENTER)

        button2 = tk.Button(button_frame, text="Abstract", command=self.run_abstract_screen, width=15, height=3, bg="#e3fef7")
        button3 = tk.Button(button_frame, text="Help", command=self.run_help_screen, width=15, height=3, bg="#e3fef7")
        button4 = tk.Button(button_frame, text="Start", command=self.run_model_screen, width=15, height=3, bg="#e3fef7")

        button2.grid(row=0, column=0, padx=(20, 20))
        button3.grid(row=0, column=1, padx=(20, 20))
        button4.grid(row=0, column=2, padx=(20, 20))

    def run_abstract_screen(self):
        self.current_screen.destroy()
        self.current_screen = tk.Toplevel(self.root)
        self.current_screen.title("Abstract Screen")
        self.current_screen.state('zoomed')

        abstract_image_path = "C:\\Users\\Asus\\OneDrive\\Desktop\\6th Sem Project\\Final Dependencies\\Abstract.jpg"
        abstract_image = Image.open(abstract_image_path)
        abstract_image = abstract_image.resize((1500, 820))
        abstract_image = ImageTk.PhotoImage(abstract_image)

        abstract_image_label = tk.Label(self.current_screen, image=abstract_image)
        abstract_image_label.image = abstract_image
        abstract_image_label.place(x=0, y=0, relwidth=1, relheight=1)

        home_button = tk.Button(
            self.current_screen,
            text="Home",
            command=self.back_to_home,
            width=10,
            height=2,
            bg="#e3fef7"
        )
        home_button.place(relx=0.02, rely=0.02)

    def run_help_screen(self):
        self.current_screen.destroy()
        self.current_screen = tk.Toplevel(self.root)
        self.current_screen.title("Help Screen")
        self.current_screen.state('zoomed')

        help_image_path = "C:\\Users\\Asus\\OneDrive\\Desktop\\This is the final folder\\Final Dependencies\\Help.png"
        help_image = Image.open(help_image_path)
        help_image = help_image.resize((1500, 820))
        help_image = ImageTk.PhotoImage(help_image)

        help_image_label = tk.Label(self.current_screen, image=help_image)
        help_image_label.image = help_image
        help_image_label.place(x=0, y=0, relwidth=1, relheight=1)

        home_button = tk.Button(
            self.current_screen,
            text="Home",
            command=self.back_to_home,
            width=10,
            height=2,
            bg="#e3fef7"
        )
        home_button.place(relx=0.02, rely=0.02)

    def run_model_screen(self):
        self.current_screen.destroy()
        ModelGUI(self.root)

    def back_to_home(self):
        self.current_screen.destroy()
        self.show_intro_screen()

class ModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Load Keras Model")
        self.root.state('zoomed')
        self.root.configure(bg='#1f497d')  # Set background color of root window
        
        # Add a label for the welcome message
        self.welcome_label = tk.Label(self.root, text="Welcome To The Disaster Detection System", font=("Times New Roman", 25), bg='#1f497d', fg='white')
        self.welcome_label.pack(pady=(20, 10))  # Adjusted pady here
        
        # Frame to hold back and next buttons
        self.navigation_frame = tk.Frame(self.root, bg='#1f497d')  # Set background color of frame
        self.navigation_frame.pack(pady=10)
        
        self.back_button = tk.Button(self.navigation_frame, text="Back", command=self.back_to_home_screen, font=("Arial", 14), height=3, width=20, bg='#e3fef7', fg='black')
        self.back_button.pack(side=tk.LEFT, padx=10)
        
        self.next_button = tk.Button(self.navigation_frame, text="Next", command=self.open_select_image_screen, font=("Arial", 14), height=3, width=20, bg='#e3fef7', fg='black')
        self.next_button.pack(side=tk.LEFT, padx=10)
        
        self.load_model_button = tk.Button(self.root, text="Load Model", command=self.load_model, font=("Arial", 14), height=3, width=20, bg='#e3fef7', fg='black')
        self.load_model_button.pack(pady=10)
        
        # Frame for model summary
        self.model_summary_frame = tk.Frame(self.root, bg='#1f497d')  
        self.model_summary_frame.pack(expand=False, fill='both', padx=20, pady=(0, 2))  # Adjusted pady here
        
        # Frame for the default screen color
        self.default_screen_frame = tk.Frame(self.root, bg='#1f497d')
        self.default_screen_frame.pack(expand=True, fill='both')
        
        # Create model summary inside self.model_summary_frame
        self.summary_frame = ttk.Frame(self.model_summary_frame, style="Custom.TFrame")  
        self.summary_frame.pack(expand=False, fill='both', padx=5, pady=3) 
        
        self.loaded_model = None  # Attribute to store the loaded model
    
    def load_model(self):
        try:
            model_path = filedialog.askopenfilename(title="Select Model", filetypes=[("Keras Model", "*.keras")])
            with open('C:\\Users\\Asus\\OneDrive\\Desktop\\6th Sem Project\\Finally Final\\modelpath.txt', 'w') as model_file:
                model_file.write(model_path)  # Write the model path to the modelpath.txt file
            self.loaded_model = load_model(model_path)
            self.show_model_summary(self.loaded_model)
            messagebox.showinfo("Model Loaded", "Model loaded successfully!")
            
            # Configure the background color of any additional space
            self.root.configure(bg='#1f497d')
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {e}")
    
    def show_model_summary(self, model):
        for widget in self.summary_frame.winfo_children():
            widget.destroy()
        
        tree = ttk.Treeview(self.summary_frame, style="Custom.Treeview")
        tree.pack(expand=True, fill='both')
        
        tree["columns"] = ("Layer", "Output Shape")
        tree.column("#0", width=0, stretch=tk.NO)
        tree.heading("#1", text="Layer")
        tree.heading("#2", text="Output Shape")
        
        for i, layer in enumerate(model.layers):
            output_shape = layer.output_shape[1:] if layer.output_shape else "-"
            tree.insert("", "end", text=f"Layer {i}", values=(layer.name, output_shape))
    
        for col in tree["columns"]:
            tree.column(col, width=150, stretch=tk.YES)
    
    def back_to_home_screen(self):
        self.root.destroy()
        root = tk.Tk()
        app = HomeScreen(root)
        root.mainloop()

    def open_select_image_screen(self):
        self.root.destroy()
        root = tk.Tk()
        app = ImagePreviewApp(root)
        root.mainloop()


class LoadingScreen:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Loading")
        self.window.state('zoomed')  # Maximize window
        self.window.configure(bg='#1f497d')  # Set background color to #1f497d

        self.label = ttk.Label(self.window, text="Loading, please wait...", font=("Arial", 45), foreground='white', background='#1f497d')  # White text color
        self.label.pack(pady=5)

    def start(self):
        self.thread = threading.Thread(target=self._update_progress)
        self.thread.start()
        self.window.mainloop()

    def _update_progress(self):
        time.sleep(8)  # Wait for 5 seconds
        self.label.config(text="Loading completed")  # Update the label to indicate task completion
        self.window.after(2000, self.window.destroy)  # Destroy the window after 2 seconds





class ImagePreviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Preview")
        self.root.state('zoomed')
        self.root.configure(bg='#1f497d')  # Set background color of root window
        
        # Add a label for the welcome message
        self.welcome_label = tk.Label(self.root, text="Welcome To The Disaster Detection System", font=("Times New Roman", 25), bg='#1f497d', fg='white')
        self.welcome_label.pack(pady=(20, 10))  # Adjusted pady here

        # Frame for buttons
        self.button_frame = tk.Frame(self.root, bg='#1f497d')
        self.button_frame.pack(side=tk.TOP, pady=10)

        # Back button
        self.back_button = tk.Button(self.button_frame, text="Back", command=self.back_to_model_from_image, font=("Arial", 16), width=15, height=2, bg='#e3fef7', fg='#1f497d')
        self.back_button.pack(side=tk.LEFT, padx=20)

        # Next button
        self.next_button = tk.Button(self.button_frame, text="Next", command=self.imagetoprocess, font=("Arial", 16), width=15, height=2, bg='#e3fef7', fg='#1f497d')
        self.next_button.pack(side=tk.LEFT, padx=20)

        # Select image button
        self.select_button = tk.Button(self.root, text="Upload Image", command=self.select_image, font=("Arial", 16), bg='#e3fef7', fg='#1f497d')
        self.select_button.pack(pady=20)

        # Image and details labels
        self.image_label = tk.Label(self.root, bg='#1f497d')
        self.image_label.pack(pady=10)

        self.details_label = tk.Label(self.root, justify='left', bg='#1f497d', fg='white', font=("Arial", 14))  # Increased font size
        self.details_label.pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Upload Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")]
        )
        if file_path:
            if self.is_image_compatible(file_path):
                # Write the image path to the imagepath.txt file
                with open('C:\\Users\\Asus\\OneDrive\\Desktop\\6th Sem Project\\Finally Final\\imagepath.txt', 'w') as path_file:
                    path_file.write(file_path)
                self.display_image_preview(file_path)
                self.display_image_details(file_path)
            else:
                messagebox.showwarning("Incompatible Image", "The selected image must have dimensions of at least 150x150 pixels.")

    def is_image_compatible(self, file_path):
        img = Image.open(file_path)
        width, height = img.size
        return width >= 150 and height >= 150

    def display_image_preview(self, file_path):
        img = Image.open(file_path)
        img = img.resize((450, 450))
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def display_image_details(self, file_path):
        file_name = os.path.basename(file_path)
        file_type = os.path.splitext(file_path)[1]
        img = Image.open(file_path)
        width, height = img.size
        details_text = f"File Name: {file_name}\nFile Path: {file_path}\nFile Type: {file_type}\nDimensions: {width} x {height}"
        self.details_label.config(text=details_text)


    def imagetoprocess(self):
        # Create an Event to signal when the ImageProcessingApp is ready
        self.app_ready_event = threading.Event()

        # Start the loading screen in a separate thread
        loading_thread = threading.Thread(target=self._start_loading_screen)
        loading_thread.start()

        # Start the ImageProcessingApp in the main thread
        self._create_image_processing_app()

    def _start_loading_screen(self):
        # Start the loading screen
        loading_screen = LoadingScreen()
        loading_screen.start()

    def _create_image_processing_app(self):
        # Create the ImageProcessingApp
        self.root.destroy()  # Assuming self.root is initialized elsewhere
        root = tk.Tk()
        app = ImageProcessingApp(root)

        # Set the flag to signal that the ImageProcessingApp is ready
        self.app_ready_event.set()

        root.mainloop()


        




    def back_to_model_from_image(self):
        self.root.destroy()
        root = tk.Tk()
        app = ModelGUI(root)
        root.mainloop()

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        self.root.state('zoomed')  # Maximize the window

        self.original_image = None
        self.processed_image = None

        # Create a frame for the buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        # Back button
        self.back_button = tk.Button(self.button_frame, text="Back", command=self.run_back_script, width=10)
        self.back_button.pack(side=tk.LEFT, padx=(10, 5))

        # Next button
        self.next_button = tk.Button(self.button_frame, text="Next", command=self.run_next_script, width=10)
        self.next_button.pack(side=tk.LEFT, padx=5)

        # Process button
        self.process_button = tk.Button(self.button_frame, text="Process Image", command=self.visualize_feature_maps, width=15)
        self.process_button.pack(side=tk.LEFT, padx=5)

        # Canvas and scrollbar
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.frame = tk.Frame(self.canvas)
        self.frame.bind("<Configure>", self.on_frame_configure)
        
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill="y")
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")
        
        self.process_steps = []
        self.current_step = 0

        # Initialize model attribute
        self.model = None

        # Automatically load the image and model
        self.load_model_and_image()
    
    def load_model_and_image(self):
        try:
            with open('C:\\Users\\Asus\\OneDrive\\Desktop\\6th Sem Project\\Finally Final\\imagepath.txt', 'r') as f:
                image_path = f.read().strip()
                if image_path:
                    self.original_image = Image.open(image_path)
                    # Resize the image to 700x700 pixels
                    resized_image = self.original_image.resize((700, 700))
                    self.display_image(resized_image)
                else:
                    messagebox.showwarning("Image Not Found", "Please specify the image path.")
        except FileNotFoundError:
            messagebox.showwarning("Image Not Found", "Please specify the image path.")

        try:
            with open('C:\\Users\\Asus\\OneDrive\\Desktop\\6th Sem Project\\Finally Final\\modelpath.txt', 'r') as f:
                model_path = f.read().strip()
                if model_path:
                    self.model = load_model(model_path)
                    self.process_button.config(state=tk.NORMAL)
                else:
                    messagebox.showwarning("Model Not Found", "Please specify the model path.")
        except FileNotFoundError:
            messagebox.showwarning("Model Not Found", "Please specify the model path.")
    
    def visualize_feature_maps(self):
        if self.original_image is not None and self.model is not None:
            self.process_button.config(state=tk.DISABLED)
            self._visualize_feature_maps()
        else:
            messagebox.showwarning("Image or Model not found", "Please make sure both image and model are loaded.")
    
    def _visualize_feature_maps(self):
        # Resize the image to match the required input size of the CNN model
        resized_image = self.original_image.resize((250, 250))
        self.display_image(resized_image, "Resized Image")
        
        # Get the convolutional and max pooling layers
        conv_pool_layers = [layer for layer in self.model.layers if isinstance(layer, (Conv2D, MaxPooling2D))]
        
        # Create a feature map model
        feature_map_model = Model(inputs=self.model.inputs, outputs=[layer.output for layer in conv_pool_layers])
        
        # Convert the image to an array
        img_array = img_to_array(resized_image)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get the feature maps for the image
        feature_maps = feature_map_model.predict(img_array)
        
        print("Number of feature maps:", len(feature_maps))
        
        # Plot feature maps of convolutional and max pooling layers
        for layer, feature_map in zip(conv_pool_layers, feature_maps):
            print("Layer:", layer.name)
            print("Feature map shape:", feature_map.shape)
            self._plot_feature_maps(layer.name, feature_map)

    def _plot_feature_maps(self, layer_name, feature_map):
        num_features = feature_map.shape[-1]
        size = feature_map.shape[1]
        images_per_row = 4  # Set the number of images to display per row
        row_count = 0
        
        # Create a frame to contain the row of images
        row_frame = tk.Frame(self.frame)
        row_frame.pack()

        for i in range(num_features):
            feature_image = feature_map[0, :, :, i]
            feature_image_mean = feature_image.mean()
            feature_image_std = feature_image.std()
            if feature_image_std != 0:  # Avoid division by zero
                feature_image -= feature_image_mean
                feature_image /= feature_image_std
            else:
                feature_image -= feature_image_mean
            feature_image *= 64
            feature_image += 128
            feature_image = np.clip(feature_image, 0, 255).astype('uint8')
            image = Image.fromarray(feature_image)

            # Display image in the row frame
            label = tk.Label(row_frame, text=f"{layer_name}_{i}")
            label.pack(side='left')
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(row_frame, image=photo)
            label.image = photo
            label.pack(side='left')  # Display images side by side
            
            row_count += 1
            
            # If the number of images in the row equals images_per_row or if this is the last image
            if row_count == images_per_row or i == num_features - 1:
                row_count = 0  # Reset row count
                # Create a new row frame
                row_frame = tk.Frame(self.frame)
                row_frame.pack()

    def _add_new_line(self):
        label = tk.Label(self.frame, text="")  # Empty label to add new line
        label.pack()

    def display_image_in_frame(self, image, layer_name):
        label = tk.Label(self.frame, text=layer_name)
        label.pack()
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self.frame, image=photo)
        label.image = photo
        label.pack(side='left')  # Display images side by side


    def display_image(self, image, layer_name=None):
        # Clear existing images from the frame
        for widget in self.frame.winfo_children():
            widget.destroy()

        # Display the new image
        image = ImageTk.PhotoImage(image)
        self.image_label = tk.Label(self.frame, image=image)
        self.image_label.image = image
        self.image_label.pack(side='top', fill='both', expand=True)  # Center the image horizontally

        # Display the layer name if provided
        if layer_name:
            label = tk.Label(self.frame, text=layer_name)
            label.pack()



    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def run_back_script(self):
        # Create the ImageProcessingApp
        self.root.destroy()  # Assuming self.root is initialized elsewhere
        root = tk.Tk()
        app = ImagePreviewApp(root)





    def run_next_script(self):
        # Create an Event to signal when the ImageProcessingApp is ready
        self.app_ready_event = threading.Event()

        # Start the loading screen in a separate thread
        loading_thread = threading.Thread(target=self._start_loading_screen)
        loading_thread.start()

        # Start the ImageProcessingApp in the main thread
        self.resultscreen()

    def _start_loading_screen(self):
        # Start the loading screen
        loading_screen = LoadingScreen()
        loading_screen.start()

    def resultscreen(self):
        # Create the ImageProcessingApp
        self.root.destroy()  # Assuming self.root is initialized elsewhere
        root = tk.Tk()
        app = ResultScreen(root)

        # Set the flag to signal that the ImageProcessingApp is ready
        self.app_ready_event.set()

        root.mainloop()
    





class ResultScreen:
    def __init__(self, root):
        self.root = root
        self.root.title("Disaster Classifier")
        self.root.configure(bg="#1f497d")
        self.root.state('zoomed')

        frame = tk.Frame(self.root, bg="#1f497d")
        frame.pack(pady=10)

        self.image_label = tk.Label(frame, bg="#1f497d")
        self.image_label.grid(row=0, column=0, padx=20, pady=10)

        self.result_label = tk.Label(frame, text="", font=("Helvetica", 16), bg="#1f497d")
        self.result_label.grid(row=0, column=1, padx=20, pady=10)

        button_frame = tk.Frame(self.root, bg="#1f497d")
        button_frame.pack(side=tk.BOTTOM, pady=(50, 10))

        back_button = tk.Button(button_frame, text="Back", command=self.run_back_script, bg="#e3fef7", fg="#1f497d", width=25, height=3)
        back_button.grid(row=0, column=0, padx=(10, 5))

        home_button = tk.Button(button_frame, text="Home", command=self.run_home_script, bg="#e3fef7", fg="#1f497d", width=25, height=3)
        home_button.grid(row=0, column=1, padx=(5, 10))
    
        self.load_and_process_image()
    @staticmethod
    def preprocess_image_for_app(file_path):
        try:
            img = load_img(file_path, target_size=(250, 250))
            img_array = img_to_array(img) / 255.0
            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def load_and_process_image(self):
        with open('C:\\Users\\Asus\\OneDrive\\Desktop\\6th Sem Project\\Finally Final\\modelpath.txt', 'r') as f:
            model_path = f.read().strip()

        with open('C:\\Users\\Asus\\OneDrive\\Desktop\\6th Sem Project\\Finally Final\\imagepath.txt', 'r') as f:
            image_path = f.read().strip()

        self.classify_and_display_result(image_path, model_path)


        
    def classify_and_display_result(self, image_path, model_path):
        model = load_model(model_path)
        target_labels = ['Fire Disaster', 'Land Disaster', 'Human', 'Nature', 'Sea', 'Rain', 'Water Disaster']
        label_encoder = LabelEncoder()
        label_encoder.fit(target_labels)
        image_array = self.preprocess_image_for_app(image_path)
        if image_array is not None:
            predictions = model.predict(np.expand_dims(image_array, axis=0))
            class_index = np.argmax(predictions[0])
            class_label = label_encoder.inverse_transform([class_index])[0]
            if class_label in ['Human', 'Nature', 'Sea', 'Rain']:
                result_text = f"Disaster Not Detected\n\n{class_label}\n"
            else:
                result_text = f"Disaster Type Prediction:\n\n{class_label}"
            self.result_label.config(text=result_text, fg="white")
            img = Image.open(image_path)
            img = img.resize((650, 650))
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img)
            self.image_label.image = img

    def run_back_script(self):
        # Create an Event to signal when the ImageProcessingApp is ready
        self.app_ready_event = threading.Event()

        # Start the loading screen in a separate thread
        loading_thread = threading.Thread(target=self.load1)
        loading_thread.start()

        # Start the ImageProcessingApp in the main thread
        self.pscreen()

    def load1(self):
        # Start the loading screen
        loading_screen = LoadingScreen()
        loading_screen.start()

    def pscreen(self):
        # Create the ImageProcessingApp
        self.root.destroy()  # Assuming self.root is initialized elsewhere
        root = tk.Tk()
        app = ImageProcessingApp(root)

        # Set the flag to signal that the ImageProcessingApp is ready
        self.app_ready_event.set()

        root.mainloop()

    def run_home_script(self):
        self.root.destroy()  # Close the current screen
        root = tk.Tk()  # Create a new Tkinter window
        app = HomeScreen(root)  # Open the home screen
        root.mainloop()

def main():
    root = tk.Tk()
    app = HomeScreen(root)
    root.mainloop()



if __name__ == "__main__":
    main()
