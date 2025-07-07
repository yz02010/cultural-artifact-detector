"""
Prediction script for the trained cultural artifact detector
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
from image_preprocessor import ImagePreprocessor

class CulturalArtifactPredictor:
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
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please make sure the model file exists and run training first.")
    
    def preprocess_image(self, image_path):
        """Preprocess an image for prediction"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing (same as training)
        image = self.preprocessor.remove_gray_black_background(image)
        image = self.preprocessor.enhance_contrast(image)
        
        # Resize and normalize
        image = cv2.resize(image, self.img_size)
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image_path, top_k=3):
        """Predict the category of an image"""
        if self.model is None:
            print("❌ Model not loaded!")
            return None
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top k predictions
            top_indices = np.argsort(predictions[0])[::-1][:top_k]
            
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
    
    def predict_batch(self, image_directory, top_k=3):
        """Predict categories for all images in a directory"""
        if not os.path.exists(image_directory):
            print(f"❌ Directory {image_directory} does not exist!")
            return
        
        image_files = [f for f in os.listdir(image_directory) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"❌ No image files found in {image_directory}")
            return
        
        print(f"🔍 Processing {len(image_files)} images...")
        
        results = {}
        for image_file in image_files[:10]:  # Limit to first 10 for demo
            image_path = os.path.join(image_directory, image_file)
            predictions = self.predict(image_path, top_k=top_k)
            
            if predictions:
                results[image_file] = predictions
                print(f"\n📸 {image_file}:")
                for pred in predictions:
                    print(f"  {pred['rank']}. {pred['category']}: {pred['confidence']:.3f}")
        
        return results

def main():
    """Main prediction function"""
    print("🎨 Cultural Artifact Detector - Prediction Tool")
    print("=" * 55)
    
    # Initialize predictor
    predictor = CulturalArtifactPredictor()
    
    if predictor.model is None:
        return
    
    # Test on sample images
    test_dir = "image_test"
    
    if os.path.exists(test_dir):
        print(f"\n🔍 Testing on images from {test_dir}...")
        predictor.predict_batch(test_dir, top_k=3)
    else:
        print(f"❌ Test directory {test_dir} not found!")
        
        # Alternative: test on a single image if provided
        print("\nTo test on a single image, use:")
        print("predictor = CulturalArtifactPredictor()")
        print("results = predictor.predict('path/to/your/image.jpg')")
    
    print("\n" + "=" * 55)
    print("✨ Prediction completed!")

if __name__ == "__main__":
    main()
