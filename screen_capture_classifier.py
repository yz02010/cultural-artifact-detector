"""
Interactive Screen Capture and Cultural Artifact Detection
This script allows you to draw a rectangle on your screen to capture an area,
then uses the trained model to classify the cultural artifact.

Dependencies: pip install pyautogui tkinter pillow
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import pyautogui
from image_preprocessor import ImagePreprocessor

class ScreenCaptureClassifier:
    def __init__(self, model_path='cultural_detector_improved.h5'):
        self.model_path = model_path
        self.model = None
        self.class_names = [
            'Bronze, Brass, and Copper', 'Carvings', 'Ceramics', 'Enamels',
            'Gold and Silver', 'Imperial Seals and Albums', 'Lacquer',
            'Other Crafts', 'Paintings', 'Sculpture', 'Textiles',
            'Timepieces and Instruments'
        ]
        self.preprocessor = ImagePreprocessor()
        self.img_size = (224, 224)
        
        # UI variables
        self.root = None
        self.canvas = None
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.screenshot = None
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            messagebox.showerror("Model Error", f"Could not load model: {e}")
    
    def take_screenshot(self):
        """Take a screenshot of the entire screen"""
        screenshot = pyautogui.screenshot()
        return np.array(screenshot)
    
    def preprocess_image(self, image_array):
        """Preprocess captured image for prediction"""
        # Convert to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image = image_array
        else:
            image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing (same as training)
        image = self.preprocessor.remove_gray_black_background(image)
        image = self.preprocessor.enhance_contrast(image)
        
        # Resize and normalize
        image = cv2.resize(image, self.img_size)
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_image(self, image_array):
        """Predict the category of the captured image"""
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
            print(f"❌ Error during prediction: {e}")
            return None
    
    def create_result_image(self, original_image, predictions, save_path="result.png"):
        """Create a result image with predictions overlay"""
        # Convert numpy array to PIL Image
        if isinstance(original_image, np.ndarray):
            pil_image = Image.fromarray(original_image)
        else:
            pil_image = original_image
        
        # Create a copy for drawing
        result_image = pil_image.copy()
        draw = ImageDraw.Draw(result_image)
        
        # Try to load a font, fallback to default if not available
        try:
            font_large = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Image dimensions
        width, height = result_image.size
        
        # Create overlay area
        overlay_height = 120
        overlay = Image.new('RGBA', (width, overlay_height), (0, 0, 0, 180))
        result_image.paste(overlay, (0, height - overlay_height), overlay)
        
        # Draw predictions
        y_offset = height - overlay_height + 10
        
        if predictions:
            # Title
            draw.text((10, y_offset), "🎨 Cultural Artifact Classification:", 
                     fill=(255, 255, 255), font=font_large)
            y_offset += 30
            
            # Top prediction (highlighted)
            top_pred = predictions[0]
            confidence_text = f"{top_pred['confidence']:.1%}"
            text = f"🏆 {top_pred['category']} ({confidence_text})"
            draw.text((10, y_offset), text, fill=(0, 255, 0), font=font_small)
            y_offset += 25
            
            # Second and third predictions
            for pred in predictions[1:3]:
                confidence_text = f"{pred['confidence']:.1%}"
                text = f"   {pred['rank']}. {pred['category']} ({confidence_text})"
                draw.text((10, y_offset), text, fill=(200, 200, 200), font=font_small)
                y_offset += 20
        else:
            draw.text((10, y_offset), "❌ Prediction failed", 
                     fill=(255, 0, 0), font=font_large)
        
        # Save the result
        result_image.save(save_path)
        print(f"💾 Result saved as {save_path}")
        
        return result_image
    
    def on_button_press(self, event):
        """Handle mouse button press"""
        self.start_x = event.x
        self.start_y = event.y
        
        # Delete previous rectangle if exists
        if self.rect_id:
            self.canvas.delete(self.rect_id)
    
    def on_mouse_drag(self, event):
        """Handle mouse drag to draw rectangle"""
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        
        # Draw rectangle
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='red', width=3
        )
    
    def on_button_release(self, event):
        """Handle mouse button release and capture the selected area"""
        if self.start_x is None or self.start_y is None:
            return
        
        # Get rectangle coordinates
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        
        # Ensure minimum size
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            messagebox.showwarning("Selection Too Small", 
                                 "Please select a larger area (minimum 50x50 pixels)")
            return
        
        # Close the selection window
        self.root.destroy()
        
        # Capture the selected area from the original screenshot
        captured_area = self.screenshot[y1:y2, x1:x2]
        
        # Show processing message
        processing_root = tk.Tk()
        processing_root.title("Processing...")
        processing_root.geometry("300x100")
        processing_label = tk.Label(processing_root, text="🔄 Analyzing artifact...", 
                                   font=("Arial", 14))
        processing_label.pack(expand=True)
        processing_root.update()
        
        # Make prediction
        predictions = self.predict_image(captured_area)
        
        # Close processing window
        processing_root.destroy()
        
        # Create and show result
        self.show_results(captured_area, predictions)
    
    def show_results(self, captured_image, predictions):
        """Display the results in a new window"""
        # Create result image with overlay
        result_image = self.create_result_image(captured_image, predictions)
        
        # Create results window
        result_window = tk.Tk()
        result_window.title("🎨 Cultural Artifact Classification Results")
        result_window.geometry("800x600")
        
        # Create main frame
        main_frame = tk.Frame(result_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image frame
        image_frame = tk.Frame(main_frame)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Resize image for display
        display_size = (400, 300)
        display_image = result_image.copy()
        display_image.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(display_image)
        
        # Display image
        image_label = tk.Label(image_frame, image=photo)
        image_label.image = photo  # Keep a reference
        image_label.pack()
        
        # Results frame
        results_frame = tk.Frame(main_frame, width=350)
        results_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        results_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(results_frame, text="🎨 Classification Results", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        if predictions:
            # Top prediction (highlighted)
            top_pred = predictions[0]
            
            # Main result frame
            main_result_frame = tk.Frame(results_frame, bg="#e8f5e8", relief=tk.RIDGE, bd=2)
            main_result_frame.pack(fill=tk.X, pady=10)
            
            tk.Label(main_result_frame, text="🏆 BEST MATCH", 
                    font=("Arial", 12, "bold"), bg="#e8f5e8").pack(pady=5)
            
            tk.Label(main_result_frame, text=top_pred['category'], 
                    font=("Arial", 14, "bold"), bg="#e8f5e8", fg="#2d5a2d").pack()
            
            tk.Label(main_result_frame, text=f"Confidence: {top_pred['confidence']:.1%}", 
                    font=("Arial", 12), bg="#e8f5e8").pack(pady=(0, 5))
            
            # Other predictions
            tk.Label(results_frame, text="Other Possibilities:", 
                    font=("Arial", 12, "bold")).pack(pady=(20, 10))
            
            for pred in predictions[1:3]:
                pred_frame = tk.Frame(results_frame, bg="#f5f5f5", relief=tk.RIDGE, bd=1)
                pred_frame.pack(fill=tk.X, pady=2)
                
                tk.Label(pred_frame, text=f"{pred['rank']}. {pred['category']}", 
                        font=("Arial", 10), bg="#f5f5f5").pack(side=tk.LEFT, padx=5)
                
                tk.Label(pred_frame, text=f"{pred['confidence']:.1%}", 
                        font=("Arial", 10), bg="#f5f5f5").pack(side=tk.RIGHT, padx=5)
        else:
            tk.Label(results_frame, text="❌ Classification Failed", 
                    font=("Arial", 14), fg="red").pack(pady=20)
        
        # Buttons frame
        button_frame = tk.Frame(results_frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
        
        # New capture button
        new_capture_btn = tk.Button(button_frame, text="📷 New Capture", 
                                   command=lambda: [result_window.destroy(), self.start_capture()],
                                   font=("Arial", 10), bg="#4CAF50", fg="white")
        new_capture_btn.pack(side=tk.LEFT, padx=5)
        
        # Close button
        close_btn = tk.Button(button_frame, text="❌ Close", 
                             command=result_window.destroy,
                             font=("Arial", 10), bg="#f44336", fg="white")
        close_btn.pack(side=tk.RIGHT, padx=5)
        
        # Print results to console
        self.print_results(predictions)
        
        result_window.mainloop()
    
    def print_results(self, predictions):
        """Print results to console"""
        print("\n" + "="*60)
        print("🎨 CULTURAL ARTIFACT CLASSIFICATION RESULTS")
        print("="*60)
        
        if predictions:
            for pred in predictions:
                icon = "🏆" if pred['rank'] == 1 else f"{pred['rank']}."
                print(f"{icon} {pred['category']}: {pred['confidence']:.3f} ({pred['confidence']:.1%})")
        else:
            print("❌ Classification failed")
        
        print("="*60)
    
    def start_capture(self):
        """Start the screen capture process"""
        if self.model is None:
            messagebox.showerror("Model Error", "Model not loaded!")
            return
        
        # Take screenshot
        self.screenshot = self.take_screenshot()
        
        # Create fullscreen transparent window for selection
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.3)
        self.root.attributes('-topmost', True)
        
        # Create canvas
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        self.canvas = tk.Canvas(self.root, width=screen_width, height=screen_height, 
                               bg='gray', highlightthickness=0)
        self.canvas.pack()
        
        # Add instructions
        instruction_text = "Draw a rectangle around the cultural artifact to classify it\nPress ESC to cancel"
        self.canvas.create_text(screen_width//2, 50, text=instruction_text, 
                               fill='white', font=('Arial', 16), justify=tk.CENTER)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_button_press)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_button_release)
        
        # Bind escape key to cancel
        self.root.bind('<Escape>', lambda e: self.root.destroy())
        
        # Focus and start
        self.root.focus_set()
        self.root.mainloop()

def main():
    """Main function to run the screen capture classifier"""
    print("🎨 Cultural Artifact Screen Capture Classifier")
    print("=" * 55)
    print("Instructions:")
    print("1. Click 'Start Capture' to begin")
    print("2. Draw a rectangle around the artifact on your screen")
    print("3. Release mouse to capture and classify")
    print("4. Press ESC to cancel selection")
    print("=" * 55)
    
    # Create classifier
    classifier = ScreenCaptureClassifier()
    
    if classifier.model is None:
        print("❌ Cannot start - model not loaded!")
        return
    
    # Create simple start window
    start_window = tk.Tk()
    start_window.title("🎨 Cultural Artifact Classifier")
    start_window.geometry("400x300")
    start_window.resizable(False, False)
    
    # Main frame
    main_frame = tk.Frame(start_window, bg="#f0f0f0")
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    # Title
    title_label = tk.Label(main_frame, text="🎨 Cultural Artifact\nClassifier", 
                          font=("Arial", 18, "bold"), bg="#f0f0f0")
    title_label.pack(pady=20)
    
    # Description
    desc_text = ("Capture any area of your screen containing\n"
                "a cultural artifact and get instant\n"
                "AI-powered classification!")
    desc_label = tk.Label(main_frame, text=desc_text, 
                         font=("Arial", 11), bg="#f0f0f0", justify=tk.CENTER)
    desc_label.pack(pady=10)
    
    # Model info
    model_info = f"✅ Model loaded: {classifier.model_path}\n📊 Categories: {len(classifier.class_names)}"
    info_label = tk.Label(main_frame, text=model_info, 
                         font=("Arial", 9), bg="#f0f0f0", fg="#666666")
    info_label.pack(pady=10)
    
    # Start button
    start_btn = tk.Button(main_frame, text="📷 Start Screen Capture", 
                         command=lambda: [start_window.withdraw(), classifier.start_capture()],
                         font=("Arial", 14, "bold"), bg="#4CAF50", fg="white",
                         width=20, height=2)
    start_btn.pack(pady=20)
    
    # Exit button
    exit_btn = tk.Button(main_frame, text="❌ Exit", 
                        command=start_window.destroy,
                        font=("Arial", 10), bg="#f44336", fg="white")
    exit_btn.pack(pady=5)
    
    start_window.mainloop()

if __name__ == "__main__":
    main()
