"""
Demo script for Cultural Artifact Detection System
This script demonstrates the complete workflow from preprocessing to prediction
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_preprocessor import ImagePreprocessor

class CulturalArtifactDemo:
    def __init__(self, data_dir="image_test", mapping_file="get_data/image_title_mapping.json"):
        self.data_dir = data_dir
        self.mapping_file = mapping_file
        self.preprocessor = ImagePreprocessor()
        
        # Load mapping data
        try:
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                self.mapping_data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Mapping file {self.mapping_file} not found")
            self.mapping_data = {}
    
    def analyze_dataset(self):
        """Analyze the dataset structure and categories"""
        print("DATASET ANALYSIS")
        print("=" * 50)
        
        if not self.mapping_data:
            print("No mapping data available")
            return
        
        # Count categories
        categories = {}
        dynasties = {}
        
        for img_id, data in self.mapping_data.items():
            category = data.get('cultural_category', 'Unknown')
            dynasty = data.get('cultural_dynasty', 'Unknown')
            
            categories[category] = categories.get(category, 0) + 1
            dynasties[dynasty] = dynasties.get(dynasty, 0) + 1
        
        print(f"Total artifacts: {len(self.mapping_data)}")
        print(f"Categories: {len(categories)}")
        print(f"Dynasties: {len(dynasties)}")
        
        print("\nTop 10 Categories:")
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        for i, (cat, count) in enumerate(sorted_categories[:10], 1):
            print(f"{i:2d}. {cat}: {count}")
        
        print("\nTop 10 Dynasties:")
        sorted_dynasties = sorted(dynasties.items(), key=lambda x: x[1], reverse=True)
        for i, (dynasty, count) in enumerate(sorted_dynasties[:10], 1):
            print(f"{i:2d}. {dynasty}: {count}")
        
        return categories, dynasties
    
    def show_sample_images(self, num_samples=6):
        """Display sample images with their preprocessing steps"""
        print(f"\nDISPLAYING {num_samples} SAMPLE IMAGES")
        print("=" * 50)
        
        if not os.path.exists(self.data_dir):
            print(f"Directory {self.data_dir} not found!")
            return
        
        image_files = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg')]
        
        if len(image_files) == 0:
            print("No image files found!")
            return
        
        # Select random samples
        np.random.seed(42)  # For reproducible results
        selected_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
        if num_samples == 1:
            axes = axes.reshape(2, 1)
        
        for i, filename in enumerate(selected_files):
            img_path = os.path.join(self.data_dir, filename)
            img_name = filename.replace('.jpg', '')
            
            # Load original image
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Preprocess image
            processed = self.preprocessor.preprocess_image(image_rgb)
            
            # Display original
            axes[0, i].imshow(image_rgb)
            axes[0, i].set_title(f'Original: {filename}', fontsize=10)
            axes[0, i].axis('off')
            
            # Display processed
            axes[1, i].imshow(processed)
            
            # Get metadata if available
            if img_name in self.mapping_data:
                metadata = self.mapping_data[img_name]
                title = f"Category: {metadata.get('cultural_category', 'Unknown')}"
                dynasty = metadata.get('cultural_dynasty', 'Unknown')
                if dynasty != 'Unknown':
                    title += f"\\nDynasty: {dynasty}"
            else:
                title = "Processed"
            
            axes[1, i].set_title(title, fontsize=10)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_images_demo.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def demonstrate_background_removal(self, num_examples=3):
        """Demonstrate background removal on specific examples"""
        print(f"\nBACKGROUND REMOVAL DEMONSTRATION")
        print("=" * 50)
        
        if not os.path.exists(self.data_dir):
            print(f"Directory {self.data_dir} not found!")
            return
        
        image_files = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg')]
        
        if len(image_files) == 0:
            print("No image files found!")
            return
        
        # Select examples
        selected_files = image_files[:num_examples]
        
        fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4 * num_examples))
        if num_examples == 1:
            axes = axes.reshape(1, 4)
        
        for i, filename in enumerate(selected_files):
            img_path = os.path.join(self.data_dir, filename)
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply preprocessing steps
            bg_removed = self.preprocessor.remove_gray_black_background(image_rgb)
            enhanced = self.preprocessor.enhance_contrast(bg_removed)
            final = self.preprocessor.preprocess_image(image_rgb)
            
            # Display results
            axes[i, 0].imshow(image_rgb)
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(bg_removed)
            axes[i, 1].set_title('Background Removed')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(enhanced)
            axes[i, 2].set_title('Contrast Enhanced')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(final)
            axes[i, 3].set_title('Final Processed')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig('background_removal_demo.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_training_visualization(self):
        """Create a visualization showing the training data distribution"""
        if not self.mapping_data:
            print("No mapping data available for visualization")
            return
        
        categories = {}
        for img_id, data in self.mapping_data.items():
            category = data.get('cultural_category', 'Unknown')
            categories[category] = categories.get(category, 0) + 1
        
        # Create bar plot
        plt.figure(figsize=(12, 8))
        categories_sorted = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        
        names = [cat for cat, _ in categories_sorted[:15]]  # Top 15 categories
        counts = [count for _, count in categories_sorted[:15]]
        
        plt.bar(range(len(names)), counts)
        plt.xlabel('Cultural Categories')
        plt.ylabel('Number of Artifacts')
        plt.title('Distribution of Cultural Artifact Categories')
        plt.xticks(range(len(names)), names, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('category_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_demo(self):
        """Run the complete demonstration"""
        print("CULTURAL ARTIFACT DETECTION SYSTEM - DEMO")
        print("=" * 60)
        
        # 1. Analyze dataset
        categories, dynasties = self.analyze_dataset()
        
        # 2. Show sample images
        self.show_sample_images(num_samples=6)
        
        # 3. Demonstrate background removal
        self.demonstrate_background_removal(num_examples=3)
        
        # 4. Create training visualization
        self.create_training_visualization()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED!")
        print("Generated files:")
        print("- sample_images_demo.png")
        print("- background_removal_demo.png")
        print("- category_distribution.png")
        print("\nTo train the model, run:")
        print("python simple_trainer.py")

def main():
    """Main demo function"""
    demo = CulturalArtifactDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()
