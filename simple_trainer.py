"""
Simplified training script for cultural artifact detection
This script focuses on the core functionality with minimal dependencies
"""

import os
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not installed. Please install it using: pip install tensorflow")
    TF_AVAILABLE = False

class SimpleCulturalDetector:
    def __init__(self, data_dir="image_test", mapping_file="get_data/image_title_mapping.json"):
        self.data_dir = data_dir
        self.mapping_file = mapping_file
        self.img_size = (224, 224)
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
    def remove_background_simple(self, image):
        """Simple background removal for gray/black backgrounds"""
        # Convert to grayscale to detect low-intensity backgrounds
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create mask for dark/gray areas (threshold can be adjusted)
        _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Convert mask to 3-channel
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        # Apply mask to original image
        result = cv2.bitwise_and(image, mask_3ch)
        
        # Replace background with white
        background = cv2.bitwise_not(mask_3ch)
        result = result + background
        
        return result
    
    def enhance_image(self, image):
        """Enhance image contrast and brightness"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to RGB
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def load_data(self):
        """Load and preprocess images"""
        print("Loading data...")
        
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        images = []
        labels = []
        
        image_files = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg')]
        print(f"Found {len(image_files)} images")
        
        for img_file in tqdm(image_files[:1000], desc="Processing"):  # Limit for demo
            img_name = img_file.replace('.jpg', '')
            
            if img_name in mapping_data:
                img_path = os.path.join(self.data_dir, img_file)
                
                try:
                    # Load image
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                        
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Remove background and enhance
                    image = self.remove_background_simple(image)
                    image = self.enhance_image(image)
                    
                    # Resize
                    image = cv2.resize(image, self.img_size)
                    
                    # Normalize
                    image = image.astype(np.float32) / 255.0
                    
                    images.append(image)
                    labels.append(mapping_data[img_name]['cultural_category'])
                    
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    continue
        
        print(f"Successfully loaded {len(images)} images")
        
        # Convert to arrays
        X = np.array(images)
        y_encoded = self.label_encoder.fit_transform(labels)
        self.class_names = self.label_encoder.classes_
        
        print(f"Classes: {self.class_names}")
        return X, y_encoded
    
    def create_simple_model(self, num_classes):
        """Create a simple CNN model"""
        if not TF_AVAILABLE:
            print("TensorFlow not available!")
            return None
            
        model = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, epochs=20):
        """Train the model"""
        if not TF_AVAILABLE:
            print("Cannot train without TensorFlow!")
            return
            
        # Load data
        X, y = self.load_data()
        
        if len(X) == 0:
            print("No data loaded!")
            return
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create model
        self.model = self.create_simple_model(len(self.class_names))
        print("\nModel Summary:")
        self.model.summary()
        
        # Train
        print("\nStarting training...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # Save model
        self.model.save('simple_cultural_detector.h5')
        print("Model saved as 'simple_cultural_detector.h5'")
        
        return history
    
    def predict(self, image_path):
        """Predict image category"""
        if self.model is None:
            print("Model not trained!")
            return None
            
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.remove_background_simple(image)
        image = self.enhance_image(image)
        image = cv2.resize(image, self.img_size)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Predict
        pred = self.model.predict(image)
        class_idx = np.argmax(pred[0])
        confidence = pred[0][class_idx]
        
        return self.class_names[class_idx], confidence

def main():
    detector = SimpleCulturalDetector()
    
    if TF_AVAILABLE:
        history = detector.train(epochs=10)  # Quick training for demo
        
        # Test prediction on first image
        test_dir = detector.data_dir
        if os.listdir(test_dir):
            test_image = os.path.join(test_dir, os.listdir(test_dir)[0])
            category, confidence = detector.predict(test_image)
            print(f"\nPrediction for {test_image}:")
            print(f"Category: {category}")
            print(f"Confidence: {confidence:.3f}")
    else:
        print("Please install TensorFlow first: pip install tensorflow")

if __name__ == "__main__":
    main()
