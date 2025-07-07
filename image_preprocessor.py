"""
Image preprocessing utilities for cultural artifact detection
This module handles background removal and image enhancement
"""

import cv2
import numpy as np
import os
from tqdm import tqdm

class ImagePreprocessor:
    def __init__(self):
        pass
    
    def remove_gray_black_background(self, image, threshold_low=30, threshold_high=80):
        """
        Remove gray and black backgrounds from cultural artifact images
        
        Args:
            image: Input image (RGB format)
            threshold_low: Lower threshold for dark areas
            threshold_high: Upper threshold for gray areas
        
        Returns:
            Image with background removed (replaced with white)
        """
        # Convert to HSV for better color separation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Convert to grayscale for intensity analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create masks for dark and gray backgrounds
        dark_mask = gray < threshold_low  # Very dark areas
        gray_mask = (gray >= threshold_low) & (gray < threshold_high)  # Gray areas
        
        # Also check for low saturation (grayish colors)
        low_sat_mask = hsv[:, :, 1] < 30  # Low saturation
        
        # Combine masks
        background_mask = dark_mask | (gray_mask & low_sat_mask)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        background_mask = background_mask.astype(np.uint8) * 255
        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)
        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel)
        
        # Create the result image
        result = image.copy()
        result[background_mask > 0] = [255, 255, 255]  # Replace background with white
        
        return result
    
    def enhance_contrast(self, image):
        """Enhance image contrast using CLAHE"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def detect_artifact_region(self, image):
        """
        Detect the main artifact region in the image
        Returns bounding box coordinates
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Find edges
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # If no contours found, return the whole image
            h, w = image.shape[:2]
            return (0, 0, w, h)
        
        # Find the largest contour (assumed to be the main artifact)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add some padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        return (x, y, w, h)
    
    def crop_to_artifact(self, image):
        """Crop image to focus on the main artifact"""
        x, y, w, h = self.detect_artifact_region(image)
        cropped = image[y:y+h, x:x+w]
        return cropped
    
    def preprocess_image(self, image, target_size=(224, 224), enhance=True, crop=True):
        """
        Complete preprocessing pipeline for cultural artifact images
        
        Args:
            image: Input image (RGB format)
            target_size: Target size for resizing
            enhance: Whether to enhance contrast
            crop: Whether to crop to artifact region
        
        Returns:
            Preprocessed image
        """
        # Remove background
        processed = self.remove_gray_black_background(image)
        
        # Enhance contrast if requested
        if enhance:
            processed = self.enhance_contrast(processed)
        
        # Crop to artifact region if requested
        if crop:
            processed = self.crop_to_artifact(processed)
        
        # Resize to target size
        processed = cv2.resize(processed, target_size)
        
        return processed
    
    def batch_preprocess(self, input_dir, output_dir, target_size=(224, 224)):
        """
        Preprocess all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
            target_size: Target size for all images
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Processing {len(image_files)} images...")
        
        for filename in tqdm(image_files, desc="Preprocessing images"):
            try:
                # Load image
                img_path = os.path.join(input_dir, filename)
                image = cv2.imread(img_path)
                
                if image is None:
                    print(f"Could not load {filename}")
                    continue
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Preprocess
                processed = self.preprocess_image(image, target_size)
                
                # Save processed image
                output_path = os.path.join(output_dir, filename)
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, processed_bgr)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        
        print(f"Preprocessing complete! Processed images saved to {output_dir}")
    
    def visualize_preprocessing(self, image_path, save_path=None):
        """
        Visualize the preprocessing steps for a single image
        """
        import matplotlib.pyplot as plt
        
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing steps
        bg_removed = self.remove_gray_black_background(image_rgb)
        enhanced = self.enhance_contrast(bg_removed)
        cropped = self.crop_to_artifact(enhanced)
        final = cv2.resize(cropped, (224, 224))
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(bg_removed)
        axes[0, 1].set_title('Background Removed')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(enhanced)
        axes[0, 2].set_title('Contrast Enhanced')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(cropped)
        axes[1, 0].set_title('Cropped to Artifact')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(final)
        axes[1, 1].set_title('Final Processed')
        axes[1, 1].axis('off')
        
        # Show artifact detection
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        axes[1, 2].imshow(edges, cmap='gray')
        axes[1, 2].set_title('Edge Detection')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def main():
    """Demo the preprocessing functionality"""
    preprocessor = ImagePreprocessor()
    
    input_dir = "image_test"
    output_dir = "processed_images"
    
    if os.path.exists(input_dir):
        # Get a sample image for visualization
        sample_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
        if sample_files:
            sample_path = os.path.join(input_dir, sample_files[0])
            print(f"Visualizing preprocessing for: {sample_path}")
            preprocessor.visualize_preprocessing(sample_path, "preprocessing_demo.png")
        
        # Batch process all images
        print(f"\nBatch processing images from {input_dir} to {output_dir}")
        preprocessor.batch_preprocess(input_dir, output_dir)
    else:
        print(f"Directory {input_dir} not found!")

if __name__ == "__main__":
    main()
