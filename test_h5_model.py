"""
Simple H5 Model Test Script
Tests the trained cultural artifact model with example images
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from image_preprocessor import ImagePreprocessor

class SimpleModelTester:
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
        """Load the trained H5 model"""
        try:
            print(f"🔄 Loading model from {self.model_path}...")
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded successfully!")
            print(f"📊 Model input shape: {self.model.input_shape}")
            print(f"🎯 Number of classes: {len(self.class_names)}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
        return True
    
    def preprocess_image(self, image_path):
        """Preprocess an image for prediction"""
        print(f"🖼️ Preprocessing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        
        print(f"   Original size: {image.shape}")
        
        # Apply background removal
        print("   🎨 Removing background...")
        image = self.preprocessor.remove_gray_black_background(image)
        
        # Apply contrast enhancement
        print("   ✨ Enhancing contrast...")
        image = self.preprocessor.enhance_contrast(image)
        
        # Resize and normalize
        print(f"   📏 Resizing to {self.img_size}...")
        image = cv2.resize(image, self.img_size)
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image, original_image
    
    def predict_image(self, image_path, show_visualization=True):
        """Predict the category of an image and show results"""
        if self.model is None:
            print("❌ Model not loaded!")
            return None
        
        try:
            # Preprocess image
            processed_image, original_image = self.preprocess_image(image_path)
            
            # Make prediction
            print("🤖 Making prediction...")
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
            
            # Display results
            self.display_results(image_path, original_image, results, show_visualization)
            
            return results
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None
    
    def display_results(self, image_path, original_image, results, show_visualization=True):
        """Display prediction results"""
        print("\n" + "="*70)
        print(f"🎨 CULTURAL ARTIFACT CLASSIFICATION RESULTS")
        print(f"📁 Image: {os.path.basename(image_path)}")
        print("="*70)
        
        if results:
            for result in results:
                icon = "🏆" if result['rank'] == 1 else f"  {result['rank']}."
                confidence_bar = "█" * int(result['confidence'] * 20)
                print(f"{icon} {result['category']:<30} {result['confidence']:>6.1%} {confidence_bar}")
            
            # Show top prediction prominently
            top_result = results[0]
            print(f"\n🎯 BEST MATCH: {top_result['category']} ({top_result['confidence']:.1%} confidence)")
        else:
            print("❌ No predictions available")
        
        print("="*70)
        
        if show_visualization:
            self.create_result_visualization(image_path, original_image, results)
    
    def create_result_visualization(self, image_path, original_image, results):
        """Create and save a visualization of the results"""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        ax1.imshow(original_image)
        ax1.set_title(f"Original Image\n{os.path.basename(image_path)}", fontsize=12)
        ax1.axis('off')
        
        # Results visualization
        if results:
            # Create a bar chart of confidence scores
            categories = [r['category'] for r in results]
            confidences = [r['confidence'] for r in results]
            colors = ['gold', 'silver', '#CD7F32']  # Gold, Silver, Bronze
            
            bars = ax2.barh(range(len(categories)), confidences, color=colors)
            ax2.set_yticks(range(len(categories)))
            ax2.set_yticklabels(categories)
            ax2.set_xlabel('Confidence Score')
            ax2.set_title('Top 3 Predictions', fontsize=12)
            ax2.set_xlim(0, 1)
            
            # Add percentage labels on bars
            for i, (bar, conf) in enumerate(zip(bars, confidences)):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{conf:.1%}', ha='left', va='center', fontweight='bold')
            
            # Highlight the top prediction
            bars[0].set_edgecolor('red')
            bars[0].set_linewidth(3)
        else:
            ax2.text(0.5, 0.5, 'No Predictions', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=16, color='red')
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = f"prediction_result_{os.path.splitext(os.path.basename(image_path))[0]}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved as: {output_path}")
        plt.show()
    
    def test_sample_images(self, test_directory="image_test", num_samples=5):
        """Test the model on sample images from a directory"""
        if not os.path.exists(test_directory):
            print(f"❌ Test directory {test_directory} not found!")
            return
        
        # Get image files
        image_files = [f for f in os.listdir(test_directory) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"❌ No image files found in {test_directory}")
            return
        
        print(f"🔍 Testing model on {min(num_samples, len(image_files))} sample images...")
        
        # Test on first few images
        for i, image_file in enumerate(image_files[:num_samples]):
            image_path = os.path.join(test_directory, image_file)
            print(f"\n{'='*50}")
            print(f"📸 Testing image {i+1}/{num_samples}: {image_file}")
            print(f"{'='*50}")
            
            results = self.predict_image(image_path, show_visualization=True)
            
            if results:
                # Save results to file
                result_file = f"test_results_{i+1}.txt"
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(f"Image: {image_file}\n")
                    f.write(f"Results:\n")
                    for result in results:
                        f.write(f"{result['rank']}. {result['category']}: {result['confidence']:.3f}\n")
                
                print(f"📄 Detailed results saved to: {result_file}")
    
    def model_info(self):
        """Display model information"""
        if self.model is None:
            print("❌ Model not loaded!")
            return
        
        print("\n🤖 MODEL INFORMATION")
        print("="*50)
        print(f"📁 Model file: {self.model_path}")
        print(f"🏗️ Architecture: {type(self.model).__name__}")
        print(f"📊 Input shape: {self.model.input_shape}")
        print(f"📊 Output shape: {self.model.output_shape}")
        print(f"🎯 Number of classes: {len(self.class_names)}")
        print(f"⚖️ Model size: {os.path.getsize(self.model_path) / (1024*1024):.1f} MB")
        
        print(f"\n🏷️ CLASSIFICATION CATEGORIES:")
        for i, category in enumerate(self.class_names, 1):
            print(f"  {i:2d}. {category}")
        print("="*50)

def main():
    """Main function to demonstrate the H5 model usage"""
    print("🎨 Cultural Artifact H5 Model Tester")
    print("="*60)
    
    # Create tester
    tester = SimpleModelTester()
    
    if tester.model is None:
        print("❌ Cannot proceed without a valid model!")
        return
    
    # Show model information
    tester.model_info()
    
    # Test on sample images
    print(f"\n🧪 TESTING MODEL PERFORMANCE")
    print("="*60)
    tester.test_sample_images(num_samples=3)
    
    print(f"\n🎉 Testing completed!")
    print(f"📁 Check the generated files:")
    print(f"   - prediction_result_*.png (visualizations)")
    print(f"   - test_results_*.txt (detailed results)")

if __name__ == "__main__":
    main()
