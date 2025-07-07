"""
Cultural Artifact Detection - Main UI Application
==================================================

A comprehensive application for cultural artifact detection with multiple modes:
1. Screen Capture & Detection - Draw rectangle on screen to capture and classify objects
2. File Upload & Detection - Upload image files for classification  
3. Batch Processing - Process multiple images from a folder
4. Camera Capture - Use webcam to capture and classify objects

Usage:
    python main_ui.py

Dependencies:
    pip install -r requirements.txt
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import pyautogui
from threading import Thread
import json
from datetime import datetime
from image_preprocessor import ImagePreprocessor
from similarity_detector import SimilarityDetector

class CulturalArtifactDetectorApp:
    def __init__(self):
        self.model_path = 'cultural_detector_improved.h5'
        self.model = None
        self.class_names = [
            'Bronze, Brass, and Copper', 'Carvings', 'Ceramics', 'Enamels',
            'Gold and Silver', 'Imperial Seals and Albums', 'Lacquer',
            'Other Crafts', 'Paintings', 'Sculpture', 'Textiles',
            'Timepieces and Instruments'
        ]
        self.preprocessor = ImagePreprocessor()
        self.img_size = (224, 224)
        
        # Similarity detector
        self.similarity_detector = None
        
        # Screen capture variables
        self.capture_root = None
        self.capture_canvas = None
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.screenshot = None
        
        # Main UI
        self.root = None
        self.setup_main_ui()
        self.load_model()
        self.initialize_similarity_detector()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.update_status(f"✅ Model loaded: {self.model_path}")
            self.enable_detection_features()
        except Exception as e:
            error_msg = f"❌ Error loading model: {e}"
            self.update_status(error_msg)
            messagebox.showerror("Model Error", f"Could not load model from {self.model_path}\n\nError: {e}")
            self.disable_detection_features()
    
    def initialize_similarity_detector(self):
        """Initialize the similarity detector in a separate thread"""
        def init_worker():
            try:
                self.root.after(0, lambda: self.update_status("🔄 Initializing similarity detector..."))
                self.similarity_detector = SimilarityDetector(feature_model='VGG16')
                
                # Check if feature database exists
                if not os.path.exists(self.similarity_detector.feature_db_file):
                    self.root.after(0, lambda: self.update_status("🔄 Building feature database (this may take a while)..."))
                    self.similarity_detector.build_feature_database()
                else:
                    # Load existing database
                    self.similarity_detector.build_feature_database(force_rebuild=False)
                
                self.root.after(0, lambda: self.update_status("✅ Similarity detector ready"))
                self.root.after(0, lambda: self.add_results("✅ Similarity detection system initialized"))
                
            except Exception as e:
                self.root.after(0, lambda e=e: self.update_status(f"❌ Similarity detector failed: {e}"))
                self.root.after(0, lambda e=e: self.add_results(f"❌ Similarity detector error: {e}"))
        
        # Start initialization in background
        thread = Thread(target=init_worker)
        thread.daemon = True
        thread.start()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.update_status(f"✅ Model loaded: {self.model_path}")
            self.enable_detection_features()
        except Exception as e:
            error_msg = f"❌ Error loading model: {e}"
            self.update_status(error_msg)
            messagebox.showerror("Model Error", f"Could not load model from {self.model_path}\n\nError: {e}")
            self.disable_detection_features()
    
    def setup_main_ui(self):
        """Create the main user interface"""
        self.root = tk.Tk()
        self.root.title("🎨 Cultural Artifact Detection Suite")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)  # Set minimum size to prevent UI cramping
        self.root.resizable(True, True)
        
        # Center the window on screen
        self.center_window()
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_container = ttk.Frame(self.root, padding="15")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=2)  # Left panel gets 2/3 weight
        main_container.columnconfigure(1, weight=3)  # Right panel gets more space
        main_container.rowconfigure(1, weight=1)     # Middle row expands
        
        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        title_label = tk.Label(header_frame, text="🎨 Cultural Artifact Detection Suite", 
                              font=("Arial", 24, "bold"), fg="#2c3e50")
        title_label.pack()
        
        subtitle_label = tk.Label(header_frame, 
                                 text="AI-Powered Classification of Cultural Artifacts", 
                                 font=("Arial", 12), fg="#7f8c8d")
        subtitle_label.pack()
        
        # Left panel - Detection modes
        left_panel = ttk.LabelFrame(main_container, text="Detection Modes", padding="12")
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 8))
        
        # Screen Capture Detection
        screen_frame = ttk.Frame(left_panel)
        screen_frame.pack(fill=tk.X, pady=(0, 12))
        
        self.screen_capture_btn = tk.Button(screen_frame, 
                                          text="📷 Screen Capture Detection",
                                          command=self.start_screen_capture,
                                          font=("Arial", 14, "bold"),
                                          bg="#3498db", fg="white",
                                          height=2, relief=tk.RAISED, bd=3)
        self.screen_capture_btn.pack(fill=tk.X)
        
        screen_desc = tk.Label(screen_frame, 
                              text="Draw a rectangle on your screen to capture\nand classify any cultural artifact",
                              font=("Arial", 9), fg="#666666", justify=tk.CENTER)
        screen_desc.pack(pady=(5, 0))
        
        # File Upload Detection
        file_frame = ttk.Frame(left_panel)
        file_frame.pack(fill=tk.X, pady=(0, 12))
        
        self.file_upload_btn = tk.Button(file_frame,
                                       text="📁 Upload Image File",
                                       command=self.upload_image_file,
                                       font=("Arial", 12),
                                       bg="#27ae60", fg="white",
                                       height=2, relief=tk.RAISED, bd=2)
        self.file_upload_btn.pack(fill=tk.X)
        
        file_desc = tk.Label(file_frame,
                            text="Select an image file from your computer\nfor artifact classification",
                            font=("Arial", 9), fg="#666666", justify=tk.CENTER)
        file_desc.pack(pady=(5, 0))
        
        # Batch Processing
        batch_frame = ttk.Frame(left_panel)
        batch_frame.pack(fill=tk.X, pady=(0, 12))
        
        self.batch_process_btn = tk.Button(batch_frame,
                                         text="📂 Batch Process Folder",
                                         command=self.batch_process_folder,
                                         font=("Arial", 12),
                                         bg="#f39c12", fg="white",
                                         height=2, relief=tk.RAISED, bd=2)
        self.batch_process_btn.pack(fill=tk.X)
        
        batch_desc = tk.Label(batch_frame,
                             text="Process multiple images from a folder\nand export results to CSV",
                             font=("Arial", 9), fg="#666666", justify=tk.CENTER)
        batch_desc.pack(pady=(5, 0))
        
        # Similarity Detection
        similarity_frame = ttk.Frame(left_panel)
        similarity_frame.pack(fill=tk.X, pady=(0, 12))
        
        self.similarity_btn = tk.Button(similarity_frame,
                                       text="🔍 Find Similar Artifacts",
                                       command=self.similarity_detection,
                                       font=("Arial", 12),
                                       bg="#e67e22", fg="white",
                                       height=2, relief=tk.RAISED, bd=2)
        self.similarity_btn.pack(fill=tk.X)
        
        similarity_desc = tk.Label(similarity_frame,
                                  text="Upload an image to find similar artifacts\nfrom the museum collection",
                                  font=("Arial", 9), fg="#666666", justify=tk.CENTER)
        similarity_desc.pack(pady=(5, 0))
        
        # Camera Capture (optional feature)
        camera_frame = ttk.Frame(left_panel)
        camera_frame.pack(fill=tk.X)
        
        self.camera_btn = tk.Button(camera_frame,
                                   text="📹 Camera Capture",
                                   command=self.camera_capture,
                                   font=("Arial", 12),
                                   bg="#9b59b6", fg="white",
                                   height=2, relief=tk.RAISED, bd=2)
        self.camera_btn.pack(fill=tk.X)
        
        camera_desc = tk.Label(camera_frame,
                              text="Use your webcam to capture\nand classify artifacts in real-time",
                              font=("Arial", 9), fg="#666666", justify=tk.CENTER)
        camera_desc.pack(pady=(5, 0))
        
        # Right panel - Results and info
        right_panel = ttk.LabelFrame(main_container, text="Information & Results", padding="12")
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Model info
        info_frame = ttk.Frame(right_panel)
        info_frame.pack(fill=tk.X, pady=(0, 12))
        
        info_title = tk.Label(info_frame, text="🤖 Model Information", 
                             font=("Arial", 12, "bold"))
        info_title.pack(anchor=tk.W)
        
        self.model_info_text = tk.Text(info_frame, height=10, wrap=tk.WORD, 
                                      font=("Courier", 9), bg="#f8f9fa", 
                                      relief=tk.SUNKEN, bd=1)
        self.model_info_text.pack(fill=tk.X, pady=(5, 0))
        
        # Results area
        results_title = tk.Label(right_panel, text="📊 Latest Results", 
                                font=("Arial", 12, "bold"))
        results_title.pack(anchor=tk.W, pady=(12, 5))
        
        # Results frame with scrollbar
        results_container = ttk.Frame(right_panel)
        results_container.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(results_container, wrap=tk.WORD, 
                                   font=("Courier", 9), bg="#f8f9fa",
                                   relief=tk.SUNKEN, bd=1)
        
        scrollbar = ttk.Scrollbar(results_container, orient=tk.VERTICAL, 
                                 command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status bar
        status_frame = ttk.Frame(main_container)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(12, 0))
        
        self.status_label = tk.Label(status_frame, text="🔄 Initializing...", 
                                    font=("Arial", 10), fg="#7f8c8d", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT)
        
        # Exit button
        exit_btn = tk.Button(status_frame, text="❌ Exit", 
                           command=self.root.quit,
                           font=("Arial", 10), bg="#e74c3c", fg="white")
        exit_btn.pack(side=tk.RIGHT)
        
        # Initialize UI state
        self.update_model_info()
        self.add_results("Welcome to Cultural Artifact Detection Suite!\n" + 
                        "Please select a detection mode to begin.\n" + 
                        "=" * 50)
    
    def center_window(self):
        """Center the main window on the screen"""
        self.root.update_idletasks()
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Set window size based on screen size (80% of screen size, but within limits)
        window_width = min(1200, int(screen_width * 0.8))
        window_height = min(800, int(screen_height * 0.8))
        
        # Ensure minimum size
        window_width = max(window_width, 1000)
        window_height = max(window_height, 700)
        
        # Calculate position to center window
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        
        # Apply the geometry
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    def update_status(self, message):
        """Update the status bar"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def update_model_info(self):
        """Update the model information display"""
        info_text = f"""Model: {self.model_path}
Categories: {len(self.class_names)}
Input Size: {self.img_size}

Supported Categories:
"""
        for i, category in enumerate(self.class_names, 1):
            info_text += f"{i:2d}. {category}\n"
        
        self.model_info_text.delete(1.0, tk.END)
        self.model_info_text.insert(1.0, info_text)
        self.model_info_text.config(state=tk.DISABLED)
    
    def add_results(self, text):
        """Add text to the results area"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {text}\n")
        self.results_text.see(tk.END)
        self.results_text.config(state=tk.DISABLED)
    
    def enable_detection_features(self):
        """Enable all detection buttons"""
        self.screen_capture_btn.config(state=tk.NORMAL)
        self.file_upload_btn.config(state=tk.NORMAL)
        self.batch_process_btn.config(state=tk.NORMAL)
        self.similarity_btn.config(state=tk.NORMAL)
        self.camera_btn.config(state=tk.NORMAL)
    
    def disable_detection_features(self):
        """Disable all detection buttons"""
        self.screen_capture_btn.config(state=tk.DISABLED)
        self.file_upload_btn.config(state=tk.DISABLED)
        self.batch_process_btn.config(state=tk.DISABLED)
        self.similarity_btn.config(state=tk.DISABLED)
        self.camera_btn.config(state=tk.DISABLED)
    
    def preprocess_image(self, image_array):
        """Preprocess image for prediction"""
        # Convert to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image = image_array
        else:
            image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing
        image = self.preprocessor.remove_gray_black_background(image)
        image = self.preprocessor.enhance_contrast(image)
        
        # Resize and normalize
        image = cv2.resize(image, self.img_size)
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_image(self, image_array):
        """Predict the category of an image"""
        if self.model is None:
            return None
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_array)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top 3 predictions
            top_indices = np.argsort(predictions[0])[::-1][:3]
            
            results = []
            for i, idx in enumerate(top_indices, 1):
                confidence = predictions[0][idx]
                category = self.class_names[idx]
                results.append({
                    'rank': i,
                    'category': category,
                    'confidence': float(confidence)
                })
            
            return results
            
        except Exception as e:
            self.add_results(f"❌ Prediction error: {e}")
            return None
    
    def start_screen_capture(self):
        """Start the screen capture detection process"""
        if self.model is None:
            messagebox.showerror("Model Error", "Model not loaded!")
            return
        
        self.add_results("📷 Starting screen capture detection...")
        self.update_status("📷 Taking screenshot...")
        
        # Hide main window
        self.root.withdraw()
        
        # Take screenshot
        try:
            self.screenshot = np.array(pyautogui.screenshot())
            self.create_capture_overlay()
        except Exception as e:
            self.add_results(f"❌ Screenshot failed: {e}")
            self.root.deiconify()
    
    def create_capture_overlay(self):
        """Create the screen capture overlay"""
        # Create fullscreen overlay window
        self.capture_root = tk.Toplevel(self.root)
        self.capture_root.attributes('-fullscreen', True)
        self.capture_root.attributes('-alpha', 0.3)
        self.capture_root.attributes('-topmost', True)
        self.capture_root.configure(bg='gray')
        
        # Create canvas
        screen_width = self.capture_root.winfo_screenwidth()
        screen_height = self.capture_root.winfo_screenheight()
        
        self.capture_canvas = tk.Canvas(self.capture_root, 
                                       width=screen_width, height=screen_height,
                                       bg='gray', highlightthickness=0)
        self.capture_canvas.pack()
        
        # Add instructions
        instruction_text = ("🎨 Cultural Artifact Detection\n\n"
                           "Draw a rectangle around the artifact to classify it\n"
                           "Press ESC to cancel")
        self.capture_canvas.create_text(screen_width//2, 80, text=instruction_text,
                                       fill='white', font=('Arial', 18, 'bold'),
                                       justify=tk.CENTER)
        
        # Bind events
        self.capture_canvas.bind('<Button-1>', self.on_capture_press)
        self.capture_canvas.bind('<B1-Motion>', self.on_capture_drag)
        self.capture_canvas.bind('<ButtonRelease-1>', self.on_capture_release)
        self.capture_root.bind('<Escape>', self.cancel_capture)
        
        # Focus
        self.capture_root.focus_set()
    
    def on_capture_press(self, event):
        """Handle mouse press for screen capture"""
        self.start_x = event.x
        self.start_y = event.y
        
        if self.rect_id:
            self.capture_canvas.delete(self.rect_id)
    
    def on_capture_drag(self, event):
        """Handle mouse drag for screen capture"""
        if self.rect_id:
            self.capture_canvas.delete(self.rect_id)
        
        self.rect_id = self.capture_canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='red', width=4
        )
    
    def on_capture_release(self, event):
        """Handle mouse release for screen capture"""
        if self.start_x is None or self.start_y is None:
            return
        
        # Get coordinates
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        
        # Check minimum size
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            self.add_results("⚠️ Selection too small, please select a larger area")
            return
        
        # Close capture overlay
        self.capture_root.destroy()
        
        # Show main window
        self.root.deiconify()
        
        # Process captured area
        self.process_screen_capture(x1, y1, x2, y2)
    
    def cancel_capture(self, event=None):
        """Cancel screen capture"""
        self.capture_root.destroy()
        self.root.deiconify()
        self.add_results("❌ Screen capture cancelled")
    
    def process_screen_capture(self, x1, y1, x2, y2):
        """Process the captured screen area"""
        self.update_status("🔄 Processing captured area...")
        
        try:
            # Extract captured area
            captured_area = self.screenshot[y1:y2, x1:x2]
            
            # Make prediction
            predictions = self.predict_image(captured_area)
            
            if predictions:
                self.display_prediction_results(captured_area, predictions, "Screen Capture")
            else:
                self.add_results("❌ Classification failed for captured area")
                
        except Exception as e:
            self.add_results(f"❌ Error processing capture: {e}")
        
        self.update_status("✅ Ready for next detection")
    
    def upload_image_file(self):
        """Upload and classify an image file"""
        if self.model is None:
            messagebox.showerror("Model Error", "Model not loaded!")
            return
        
        # File dialog
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        self.add_results(f"📁 Processing uploaded file: {os.path.basename(file_path)}")
        self.update_status("🔄 Processing uploaded image...")
        
        try:
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                self.add_results("❌ Could not load image file")
                return
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Make prediction
            predictions = self.predict_image(image_rgb)
            
            if predictions:
                self.display_prediction_results(image_rgb, predictions, "File Upload")
            else:
                self.add_results("❌ Classification failed for uploaded image")
                
        except Exception as e:
            self.add_results(f"❌ Error processing file: {e}")
        
        self.update_status("✅ Ready for next detection")
    
    def batch_process_folder(self):
        """Batch process images from a folder"""
        if self.model is None:
            messagebox.showerror("Model Error", "Model not loaded!")
            return
        
        # Folder dialog
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if not folder_path:
            return
        
        self.add_results(f"📂 Starting batch processing: {folder_path}")
        self.update_status("🔄 Batch processing in progress...")
        
        # Start batch processing in separate thread
        thread = Thread(target=self._batch_process_worker, args=(folder_path,))
        thread.daemon = True
        thread.start()
    
    def _batch_process_worker(self, folder_path):
        """Worker function for batch processing"""
        try:
            # Get image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            image_files = []
            
            for file in os.listdir(folder_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(folder_path, file))
            
            if not image_files:
                self.add_results("❌ No image files found in selected folder")
                return
            
            self.add_results(f"📊 Found {len(image_files)} images to process")
            
            # Process each image
            results = []
            for i, file_path in enumerate(image_files, 1):
                try:
                    # Update progress
                    self.root.after(0, lambda i=i, total=len(image_files): 
                                   self.update_status(f"🔄 Processing {i}/{total}..."))
                    
                    # Load and predict
                    image = cv2.imread(file_path)
                    if image is not None:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        predictions = self.predict_image(image_rgb)
                        
                        if predictions:
                            results.append({
                                'file': os.path.basename(file_path),
                                'top_category': predictions[0]['category'],
                                'confidence': predictions[0]['confidence'],
                                'all_predictions': predictions
                            })
                            
                            self.root.after(0, lambda f=os.path.basename(file_path), 
                                          cat=predictions[0]['category'],
                                          conf=predictions[0]['confidence']:
                                          self.add_results(f"✅ {f}: {cat} ({conf:.1%})"))
                        else:
                            self.root.after(0, lambda f=os.path.basename(file_path):
                                          self.add_results(f"❌ Failed: {f}"))
                    
                except Exception as e:
                    self.root.after(0, lambda f=os.path.basename(file_path), e=e:
                                   self.add_results(f"❌ Error processing {f}: {e}"))
            
            # Save results to CSV
            if results:
                self._save_batch_results(results, folder_path)
            
            self.root.after(0, lambda: self.update_status("✅ Batch processing completed"))
            self.root.after(0, lambda: self.add_results(f"🎉 Batch processing completed! Processed {len(results)} images"))
            
        except Exception as e:
            self.root.after(0, lambda e=e: self.add_results(f"❌ Batch processing error: {e}"))
            self.root.after(0, lambda: self.update_status("❌ Batch processing failed"))
    
    def _save_batch_results(self, results, folder_path):
        """Save batch processing results to CSV"""
        try:
            import csv
            
            output_file = os.path.join(folder_path, f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['filename', 'top_category', 'confidence', 'rank_2_category', 'rank_2_confidence', 
                             'rank_3_category', 'rank_3_confidence']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in results:
                    preds = result['all_predictions']
                    row = {
                        'filename': result['file'],
                        'top_category': preds[0]['category'],
                        'confidence': f"{preds[0]['confidence']:.4f}",
                        'rank_2_category': preds[1]['category'] if len(preds) > 1 else '',
                        'rank_2_confidence': f"{preds[1]['confidence']:.4f}" if len(preds) > 1 else '',
                        'rank_3_category': preds[2]['category'] if len(preds) > 2 else '',
                        'rank_3_confidence': f"{preds[2]['confidence']:.4f}" if len(preds) > 2 else ''
                    }
                    writer.writerow(row)
            
            self.root.after(0, lambda f=output_file: 
                           self.add_results(f"💾 Results saved to: {os.path.basename(f)}"))
            
        except Exception as e:
            self.root.after(0, lambda e=e: 
                           self.add_results(f"❌ Error saving results: {e}"))
    
    def camera_capture(self):
        """Camera capture feature (placeholder)"""
        messagebox.showinfo("Feature Coming Soon", 
                           "Camera capture feature will be implemented in a future update!")
        self.add_results("📹 Camera capture feature requested (coming soon)")
    
    def similarity_detection(self):
        """Find similar artifacts from the image library"""
        if self.similarity_detector is None:
            messagebox.showerror("Feature Not Ready", 
                               "Similarity detection system is not ready yet.\nPlease wait for initialization to complete.")
            return
        
        # File dialog
        file_path = filedialog.askopenfilename(
            title="Select Image to Find Similar Artifacts",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        self.add_results(f"🔍 Finding similar artifacts for: {os.path.basename(file_path)}")
        self.update_status("🔄 Searching for similar artifacts...")
        
        try:
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                self.add_results("❌ Could not load image file")
                return
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find similar artifacts
            similar_results = self.similarity_detector.find_similar_artifacts(image_rgb, top_k=5)
            
            if similar_results:
                self.display_similarity_results(image_rgb, similar_results, "Similarity Detection")
            else:
                self.add_results("❌ No similar artifacts found")
                
        except Exception as e:
            self.add_results(f"❌ Error during similarity detection: {e}")
        
        self.update_status("✅ Ready for next detection")
    
    def display_prediction_results(self, image_array, predictions, source):
        """Display prediction results in a popup window"""
        # Create results window
        result_window = tk.Toplevel(self.root)
        result_window.title(f"🎨 Classification Results - {source}")
        result_window.geometry("900x700")
        result_window.resizable(True, True)
        
        # Main frame
        main_frame = ttk.Frame(result_window, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Image frame
        image_frame = ttk.LabelFrame(main_frame, text="Captured Image", padding="10")
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Display image
        self._display_image_in_frame(image_frame, image_array, max_size=(500, 400))
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Classification Results", padding="10")
        results_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Display results
        self._display_results_in_frame(results_frame, predictions, result_window)
        
        # Add to main results log
        if predictions:
            top_pred = predictions[0]
            self.add_results(f"🎯 {source} Result: {top_pred['category']} ({top_pred['confidence']:.1%})")
    
    def _display_results_in_frame(self, frame, predictions, parent_window):
        """Display prediction results in the given frame"""
        if not predictions:
            tk.Label(frame, text="❌ Classification Failed", 
                    font=("Arial", 14), fg="red").pack(pady=20)
            return
        
        # Top prediction
        top_pred = predictions[0]
        
        top_frame = tk.Frame(frame, bg="#e8f5e8", relief=tk.RIDGE, bd=2)
        top_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(top_frame, text="🏆 BEST MATCH", 
                font=("Arial", 12, "bold"), bg="#e8f5e8").pack(pady=5)
        
        tk.Label(top_frame, text=top_pred['category'], 
                font=("Arial", 14, "bold"), bg="#e8f5e8", fg="#2d5a2d").pack()
        
        tk.Label(top_frame, text=f"Confidence: {top_pred['confidence']:.1%}", 
                font=("Arial", 12), bg="#e8f5e8").pack(pady=(0, 5))
        
        # Other predictions
        tk.Label(frame, text="Other Possibilities:", 
                font=("Arial", 12, "bold")).pack(pady=(20, 10))
        
        for pred in predictions[1:3]:
            pred_frame = tk.Frame(frame, bg="#f5f5f5", relief=tk.RIDGE, bd=1)
            pred_frame.pack(fill=tk.X, pady=2)
            
            tk.Label(pred_frame, text=f"{pred['rank']}. {pred['category']}", 
                    font=("Arial", 10), bg="#f5f5f5").pack(side=tk.LEFT, padx=5)
            
            tk.Label(pred_frame, text=f"{pred['confidence']:.1%}", 
                    font=("Arial", 10), bg="#f5f5f5").pack(side=tk.RIGHT, padx=5)
        
        # Buttons
        button_frame = tk.Frame(frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
        
        tk.Button(button_frame, text="❌ Close", 
                 command=parent_window.destroy,
                 font=("Arial", 10), bg="#e74c3c", fg="white").pack(fill=tk.X)
    
    def display_similarity_results(self, query_image, similarity_results, source):
        """Display similarity detection results in a popup window"""
        # Create results window
        result_window = tk.Toplevel(self.root)
        result_window.title(f"🔍 Similarity Results - {source}")
        result_window.geometry("1200x800")
        result_window.resizable(True, True)
        
        # Main frame with scrollable content
        main_frame = ttk.Frame(result_window, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Query image frame (left side)
        query_frame = ttk.LabelFrame(main_frame, text="Query Image", padding="10")
        query_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Display query image
        self._display_image_in_frame(query_frame, query_image, max_size=(300, 300))
        
        # Results frame (right side)
        results_frame = ttk.LabelFrame(main_frame, text="Similar Artifacts", padding="10")
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create scrollable frame for results
        canvas = tk.Canvas(results_frame)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Display similarity results
        for i, result in enumerate(similarity_results, 1):
            # Create frame for each result
            result_frame = tk.Frame(scrollable_frame, relief=tk.RIDGE, bd=2, padx=10, pady=10)
            result_frame.pack(fill=tk.X, pady=5)
            
            # Rank and similarity score
            header_frame = tk.Frame(result_frame)
            header_frame.pack(fill=tk.X, pady=(0, 5))
            
            rank_label = tk.Label(header_frame, text=f"#{i}", 
                                 font=("Arial", 14, "bold"), fg="#2c3e50")
            rank_label.pack(side=tk.LEFT)
            
            similarity_label = tk.Label(header_frame, 
                                       text=f"Similarity: {result['similarity']:.1%}",
                                       font=("Arial", 12, "bold"), fg="#27ae60")
            similarity_label.pack(side=tk.RIGHT)
            
            # Content frame with image and details
            content_frame = tk.Frame(result_frame)
            content_frame.pack(fill=tk.X)
            
            # Load and display artifact image
            try:
                if os.path.exists(result['image_path']):
                    artifact_image = cv2.imread(result['image_path'])
                    if artifact_image is not None:
                        artifact_image_rgb = cv2.cvtColor(artifact_image, cv2.COLOR_BGR2RGB)
                        
                        # Image frame
                        img_frame = tk.Frame(content_frame)
                        img_frame.pack(side=tk.LEFT, padx=(0, 10))
                        
                        self._display_image_in_frame(img_frame, artifact_image_rgb, max_size=(150, 150))
                
            except Exception as e:
                print(f"Error loading image {result['image_path']}: {e}")
            
            # Details frame
            details_frame = tk.Frame(content_frame)
            details_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Title and details
            title_label = tk.Label(details_frame, text=result['title'], 
                                  font=("Arial", 12, "bold"), fg="#2c3e50",
                                  wraplength=400, justify=tk.LEFT)
            title_label.pack(anchor=tk.W, pady=(0, 5))
            
            # Cultural information
            info_text = f"Category: {result['cultural_category']}\n"
            info_text += f"Dynasty: {result['cultural_dynasty']}\n"
            info_text += f"Cultural Number: {result['cultural_number']}\n"
            info_text += f"Filename: {result['filename']}"
            
            info_label = tk.Label(details_frame, text=info_text, 
                                 font=("Arial", 9), fg="#666666",
                                 justify=tk.LEFT, anchor=tk.W)
            info_label.pack(anchor=tk.W)
        
        # Pack scrollable components
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Control buttons frame
        button_frame = tk.Frame(result_window)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=10)
        
        # Close button
        close_btn = tk.Button(button_frame, text="❌ Close", 
                             command=result_window.destroy,
                             font=("Arial", 12), bg="#e74c3c", fg="white")
        close_btn.pack(side=tk.RIGHT, padx=5)
        
        # New search button
        new_search_btn = tk.Button(button_frame, text="🔍 New Search", 
                                  command=lambda: [result_window.destroy(), self.similarity_detection()],
                                  font=("Arial", 12), bg="#3498db", fg="white")
        new_search_btn.pack(side=tk.RIGHT, padx=5)
        
        # Add to main results log
        if similarity_results:
            top_result = similarity_results[0]
            self.add_results(f"🔍 Similarity Result: {top_result['title']} ({top_result['similarity']:.1%})")
    
    def _display_image_in_frame(self, frame, image_array, max_size=(500, 400)):
        """Display image in the given frame with specified max size"""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image_array)
            
            # Resize for display while maintaining aspect ratio
            pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Display
            image_label = tk.Label(frame, image=photo)
            image_label.image = photo  # Keep reference
            image_label.pack()
            
        except Exception as e:
            error_label = tk.Label(frame, text=f"Error displaying image: {e}")
            error_label.pack()
    
    def run(self):
        """Run the main application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\n👋 Application closed by user")

def main():
    """Main function"""
    print("🎨 Cultural Artifact Detection Suite")
    print("=" * 40)
    
    app = CulturalArtifactDetectorApp()
    app.run()

if __name__ == "__main__":
    main()
