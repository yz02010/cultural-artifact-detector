"""
Improved training script for cultural artifact detection that handles imbalanced datasets
"""

import os
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
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

class ImprovedCulturalDetector:
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
    
    def filter_categories(self, labels, min_samples=2):
        """Filter out categories with too few samples"""
        label_counts = Counter(labels)
        valid_labels = [label for label, count in label_counts.items() if count >= min_samples]
        
        print(f"Filtering categories with less than {min_samples} samples...")
        print(f"Categories before filtering: {len(label_counts)}")
        print(f"Categories after filtering: {len(valid_labels)}")
        
        # Show filtered categories
        filtered_out = [label for label, count in label_counts.items() if count < min_samples]
        if filtered_out:
            print("Filtered out categories:")
            for label in filtered_out:
                print(f"  - {label}: {label_counts[label]} samples")
        
        return valid_labels
    
    def load_data(self, max_samples=None):
        """Load and preprocess images with improved filtering"""
        print("Loading data...")
        
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        images = []
        labels = []
        
        image_files = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg')]
        
        # Limit samples if specified
        if max_samples:
            image_files = image_files[:max_samples]
        
        print(f"Found {len(image_files)} images")
        
        for img_file in tqdm(image_files, desc="Processing"):
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
        
        # Filter categories with too few samples
        valid_categories = self.filter_categories(labels, min_samples=2)
        
        # Keep only valid samples
        filtered_images = []
        filtered_labels = []
        
        for img, label in zip(images, labels):
            if label in valid_categories:
                filtered_images.append(img)
                filtered_labels.append(label)
        
        print(f"After filtering: {len(filtered_images)} images from {len(valid_categories)} categories")
        
        # Convert to arrays
        X = np.array(filtered_images)
        y_encoded = self.label_encoder.fit_transform(filtered_labels)
        self.class_names = self.label_encoder.classes_
        
        # Show final class distribution
        print("\nFinal class distribution:")
        unique, counts = np.unique(y_encoded, return_counts=True)
        for class_idx, count in zip(unique, counts):
            print(f"  {self.class_names[class_idx]}: {count} samples")
        
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
    
    def create_balanced_model(self, num_classes):
        """Create a model better suited for imbalanced data"""
        if not TF_AVAILABLE:
            print("TensorFlow not available!")
            return None
        
        # Use a pre-trained base model for better performance
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model, base_model
    
    def calculate_class_weights(self, y):
        """Calculate class weights to handle imbalanced data"""
        unique, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        class_weights = {}
        for class_idx, count in zip(unique, counts):
            # Inverse frequency weighting
            weight = total_samples / (len(unique) * count)
            class_weights[class_idx] = weight
        
        print("Class weights:")
        for class_idx, weight in class_weights.items():
            print(f"  {self.class_names[class_idx]}: {weight:.2f}")
        
        return class_weights
    
    def train(self, epochs=20, use_transfer_learning=True):
        """Train the model with improved handling for imbalanced data"""
        if not TF_AVAILABLE:
            print("Cannot train without TensorFlow!")
            return
            
        # Load data
        X, y = self.load_data(max_samples=500)  # Limit for faster training
        
        if len(X) == 0:
            print("No data loaded!")
            return
        
        # Check if we have enough data for validation split
        min_class_count = min(Counter(y).values())
        if min_class_count < 2:
            print("Not enough samples per class for train/validation split!")
            return
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(y)
        
        # Split data - use stratify only if we have enough samples
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # If stratified split fails, use random split
            print("Using random split (not stratified)")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Create model
        if use_transfer_learning and len(self.class_names) > 5:
            self.model, base_model = self.create_balanced_model(len(self.class_names))
            print("Using transfer learning model (MobileNetV2)")
        else:
            self.model = self.create_simple_model(len(self.class_names))
            print("Using simple CNN model")
        
        print("\nModel Summary:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        # Train
        print("\nStarting training...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=16,  # Smaller batch size for better training
            epochs=epochs,
            validation_data=(X_val, y_val),
            class_weight=class_weights,  # Handle imbalanced data
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tune if using transfer learning
        if use_transfer_learning and len(self.class_names) > 5:
            print("\nFine-tuning model...")
            base_model.trainable = True
            
            # Use lower learning rate for fine-tuning
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train for a few more epochs
            fine_tune_history = self.model.fit(
                X_train, y_train,
                batch_size=16,
                epochs=5,
                validation_data=(X_val, y_val),
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine histories
            for key in history.history:
                history.history[key].extend(fine_tune_history.history[key])
        
        # Save model
        self.model.save('cultural_detector_improved.h5')
        print("Model saved as 'cultural_detector_improved.h5'")
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        """Plot training metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history_improved.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict(self, image_path, top_k=3):
        """Predict image category with confidence scores"""
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
        pred = self.model.predict(image, verbose=0)
        
        # Get top k predictions
        top_indices = np.argsort(pred[0])[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'category': self.class_names[idx],
                'confidence': float(pred[0][idx])
            })
        
        return results

def main():
    detector = ImprovedCulturalDetector()
    
    if TF_AVAILABLE:
        print("Starting improved training...")
        history = detector.train(epochs=15, use_transfer_learning=True)
        
        # Test prediction on first image
        test_dir = detector.data_dir
        if os.listdir(test_dir):
            test_image = os.path.join(test_dir, os.listdir(test_dir)[0])
            results = detector.predict(test_image, top_k=3)
            
            print(f"\nTop 3 predictions for {test_image}:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['category']}: {result['confidence']:.3f}")
    else:
        print("Please install TensorFlow first: pip install tensorflow")

if __name__ == "__main__":
    main()
