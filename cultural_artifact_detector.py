import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from PIL import Image
import albumentations as A
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CulturalArtifactDetector:
    def __init__(self, data_dir="image_test", mapping_file="get_data/image_title_mapping.json", 
                 img_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.mapping_file = mapping_file
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
        # Image preprocessing pipeline to handle gray/black backgrounds
        self.transform = A.Compose([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.8),  # Enhance contrast
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def remove_background(self, image):
        """Remove gray and black backgrounds using color-based masking"""
        # Convert to HSV for better color filtering
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define range for gray and black colors
        # Gray range
        gray_lower = np.array([0, 0, 0])
        gray_upper = np.array([180, 30, 100])
        
        # Create mask for gray/black areas
        gray_mask = cv2.inRange(hsv, gray_lower, gray_upper)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)
        
        # Invert mask to get foreground
        object_mask = cv2.bitwise_not(gray_mask)
        
        # Apply mask to original image
        result = cv2.bitwise_and(image, image, mask=object_mask)
        
        # Replace background with white
        result[gray_mask > 0] = [255, 255, 255]
        
        return result
    
    def load_and_preprocess_data(self):
        """Load images and labels from the dataset"""
        print("Loading and preprocessing data...")
        
        # Load mapping data
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        images = []
        labels = []
        categories = []
        
        # Get list of available image files
        image_files = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg')]
        
        print(f"Found {len(image_files)} images in {self.data_dir}")
        
        for img_file in tqdm(image_files, desc="Processing images"):
            img_name = img_file.replace('.jpg', '')
            
            if img_name in mapping_data:
                img_path = os.path.join(self.data_dir, img_file)
                
                try:
                    # Load image
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                        
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Remove background
                    image = self.remove_background(image)
                    
                    # Resize image
                    image = cv2.resize(image, self.img_size)
                    
                    # Apply augmentations
                    transformed = self.transform(image=image)
                    image = transformed['image']
                    
                    images.append(image)
                    
                    # Use cultural_category as label
                    category = mapping_data[img_name]['cultural_category']
                    categories.append(category)
                    labels.append(img_name)
                    
                except Exception as e:
                    print(f"Error processing {img_file}: {str(e)}")
                    continue
        
        print(f"Successfully processed {len(images)} images")
        
        # Convert to numpy arrays
        X = np.array(images, dtype=np.float32)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(categories)
        self.class_names = self.label_encoder.classes_
        
        print(f"Found {len(self.class_names)} categories: {self.class_names}")
        
        return X, y_encoded, labels, categories
    
    def create_model(self, num_classes):
        """Create a CNN model for artifact classification"""
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Fine-tune the last few layers
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        return model
    
    def train_model(self, X, y, validation_split=0.2, epochs=50):
        """Train the model"""
        print("Starting model training...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Create model
        num_classes = len(self.class_names)
        self.model = self.create_model(num_classes)
        
        print("Model architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_cultural_artifact_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def plot_training_history(self, history):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-3 Accuracy
        axes[1, 0].plot(history.history['top_3_accuracy'], label='Training Top-3 Accuracy')
        axes[1, 0].plot(history.history['val_top_3_accuracy'], label='Validation Top-3 Accuracy')
        axes[1, 0].set_title('Top-3 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-3 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate (if available)
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nHistory Not Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_image(self, image_path, top_k=3):
        """Predict the category of a single image"""
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Remove background
        image = self.remove_background(image)
        
        # Resize and normalize
        image = cv2.resize(image, self.img_size)
        transformed = self.transform(image=image)
        image = transformed['image']
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        # Predict
        predictions = self.model.predict(image)
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        
        results = []
        for i in top_indices:
            results.append({
                'category': self.class_names[i],
                'confidence': float(predictions[0][i])
            })
        
        return results
    
    def save_model(self, model_path='cultural_artifact_detector.h5'):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(model_path)
            
            # Save label encoder
            import pickle
            with open('label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            print(f"Model saved to {model_path}")
            print("Label encoder saved to label_encoder.pkl")
    
    def load_model(self, model_path='cultural_artifact_detector.h5'):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(model_path)
        
        # Load label encoder
        import pickle
        with open('label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.class_names = self.label_encoder.classes_
        print(f"Model loaded from {model_path}")

def main():
    """Main training function"""
    print("Cultural Artifact Detection System")
    print("==================================")
    
    # Initialize detector
    detector = CulturalArtifactDetector()
    
    # Load and preprocess data
    X, y, labels, categories = detector.load_and_preprocess_data()
    
    if len(X) == 0:
        print("No data found! Please check your image directory and mapping file.")
        return
    
    # Print dataset statistics
    unique_categories, counts = np.unique(categories, return_counts=True)
    print("\nDataset Statistics:")
    print("-" * 50)
    for cat, count in zip(unique_categories, counts):
        print(f"{cat}: {count} images")
    
    # Train model
    history = detector.train_model(X, y, epochs=50)
    
    # Plot training history
    detector.plot_training_history(history)
    
    # Save model
    detector.save_model()
    
    # Example prediction
    if len(os.listdir(detector.data_dir)) > 0:
        sample_image = os.path.join(detector.data_dir, os.listdir(detector.data_dir)[0])
        print(f"\nSample prediction for {sample_image}:")
        results = detector.predict_image(sample_image)
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['category']}: {result['confidence']:.3f}")

if __name__ == "__main__":
    main()
