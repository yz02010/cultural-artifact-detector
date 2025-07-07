"""
Cultural Artifact Similarity Detector
====================================

This module creates a feature-based similarity detection system that:
1. Extracts features from images using a pre-trained CNN
2. Builds a feature database from the image library
3. Finds the most similar artifacts based on shape and material
4. Returns matched items with titles from image_title_mapping.json

Usage:
    similarity_detector = SimilarityDetector()
    similarity_detector.build_feature_database()
    results = similarity_detector.find_similar_artifacts(query_image)
"""

import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pickle
from datetime import datetime
from image_preprocessor import ImagePreprocessor

class SimilarityDetector:
    def __init__(self, feature_model='VGG16'):
        """
        Initialize the similarity detector
        
        Args:
            feature_model: 'VGG16' or 'ResNet50' for feature extraction
        """
        self.feature_model_name = feature_model
        self.feature_model = None
        self.feature_database = {}
        self.image_titles = {}
        self.pca = None
        self.img_size = (224, 224)
        self.preprocessor = ImagePreprocessor()
        
        # Paths
        self.image_folder = 'image'
        self.mapping_file = 'get_data/image_title_mapping.json'
        self.feature_db_file = 'feature_database.pkl'
        
        # Load components
        self.load_feature_model()
        self.load_image_titles()
    
    def load_feature_model(self):
        """Load pre-trained feature extraction model"""
        try:
            if self.feature_model_name == 'VGG16':
                # Use VGG16 without top layers for feature extraction
                self.feature_model = VGG16(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg',
                    input_shape=(224, 224, 3)
                )
                self.preprocess_func = vgg_preprocess
            elif self.feature_model_name == 'ResNet50':
                # Use ResNet50 without top layers for feature extraction
                self.feature_model = ResNet50(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg',
                    input_shape=(224, 224, 3)
                )
                self.preprocess_func = resnet_preprocess
            else:
                raise ValueError("Unsupported model. Use 'VGG16' or 'ResNet50'")
            
            print(f"✅ Feature extraction model loaded: {self.feature_model_name}")
            
        except Exception as e:
            print(f"❌ Error loading feature model: {e}")
            self.feature_model = None
    
    def load_image_titles(self):
        """Load image title mapping from JSON file"""
        try:
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                self.image_titles = json.load(f)
            print(f"✅ Loaded {len(self.image_titles)} image titles from mapping file")
        except Exception as e:
            print(f"❌ Error loading image titles: {e}")
            self.image_titles = {}
    
    def extract_features(self, image_array):
        """
        Extract features from an image using the pre-trained model
        
        Args:
            image_array: numpy array of the image (RGB format)
            
        Returns:
            numpy array of extracted features
        """
        if self.feature_model is None:
            return None
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_array)
            
            # Extract features
            features = self.feature_model.predict(processed_image, verbose=0)
            
            # Flatten if needed
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)
            
            return features[0]  # Return single feature vector
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return None
    
    def preprocess_image(self, image_array):
        """Preprocess image for feature extraction"""
        # Apply background removal and enhancement
        image = self.preprocessor.remove_gray_black_background(image_array)
        image = self.preprocessor.enhance_contrast(image)
        
        # Resize to model input size
        image = cv2.resize(image, self.img_size)
        
        # Convert to float and add batch dimension
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)
        
        # Apply model-specific preprocessing
        image = self.preprocess_func(image)
        
        return image
    
    def build_feature_database(self, force_rebuild=False):
        """
        Build feature database from all images in the image folder
        
        Args:
            force_rebuild: If True, rebuild even if database exists
        """
        # Check if database already exists
        if not force_rebuild and os.path.exists(self.feature_db_file):
            print(f"📦 Loading existing feature database: {self.feature_db_file}")
            try:
                with open(self.feature_db_file, 'rb') as f:
                    data = pickle.load(f)
                    self.feature_database = data['features']
                    self.pca = data.get('pca', None)
                print(f"✅ Loaded feature database with {len(self.feature_database)} images")
                return
            except Exception as e:
                print(f"❌ Error loading existing database: {e}")
                print("🔄 Building new feature database...")
        
        print("🔄 Building feature database from image library...")
        
        if not os.path.exists(self.image_folder):
            print(f"❌ Image folder not found: {self.image_folder}")
            return
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        
        for file in os.listdir(self.image_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        
        if not image_files:
            print(f"❌ No image files found in {self.image_folder}")
            return
        
        print(f"📊 Found {len(image_files)} images to process")
        
        # Extract features for each image
        features_list = []
        valid_images = []
        
        for i, filename in enumerate(image_files, 1):
            try:
                if i % 50 == 0:
                    print(f"🔄 Processing image {i}/{len(image_files)}: {filename}")
                
                # Load image
                image_path = os.path.join(self.image_folder, filename)
                image = cv2.imread(image_path)
                
                if image is None:
                    continue
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Extract features
                features = self.extract_features(image_rgb)
                
                if features is not None:
                    self.feature_database[filename] = features
                    features_list.append(features)
                    valid_images.append(filename)
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                continue
        
        print(f"✅ Successfully processed {len(valid_images)} images")
        
        # Apply PCA for dimensionality reduction (optional)
        if len(features_list) > 0:
            features_array = np.array(features_list)
            
            # Use PCA to reduce dimensionality (keep 95% variance)
            self.pca = PCA(n_components=0.95, random_state=42)
            reduced_features = self.pca.fit_transform(features_array)
            
            print(f"📉 Reduced feature dimensions from {features_array.shape[1]} to {reduced_features.shape[1]}")
            
            # Update feature database with reduced features
            for i, filename in enumerate(valid_images):
                self.feature_database[filename] = reduced_features[i]
        
        # Save feature database
        self.save_feature_database()
    
    def save_feature_database(self):
        """Save feature database to file"""
        try:
            data = {
                'features': self.feature_database,
                'pca': self.pca,
                'model_name': self.feature_model_name,
                'created_at': datetime.now().isoformat()
            }
            
            with open(self.feature_db_file, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"💾 Feature database saved: {self.feature_db_file}")
            
        except Exception as e:
            print(f"❌ Error saving feature database: {e}")
    
    def find_similar_artifacts(self, query_image, top_k=5):
        """
        Find most similar artifacts to the query image
        
        Args:
            query_image: numpy array of the query image (RGB format)
            top_k: number of most similar items to return
            
        Returns:
            list of dictionaries with similarity results
        """
        if not self.feature_database:
            print("❌ Feature database is empty. Please build it first.")
            return []
        
        # Extract features from query image
        query_features = self.extract_features(query_image)
        if query_features is None:
            return []
        
        # Apply PCA if available
        if self.pca is not None:
            query_features = self.pca.transform(query_features.reshape(1, -1))[0]
        
        # Calculate similarities with all images in database
        similarities = []
        
        for filename, db_features in self.feature_database.items():
            try:
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    query_features.reshape(1, -1),
                    db_features.reshape(1, -1)
                )[0][0]
                
                similarities.append({
                    'filename': filename,
                    'similarity': float(similarity),
                    'distance': 1.0 - similarity  # Convert to distance
                })
                
            except Exception as e:
                print(f"❌ Error calculating similarity for {filename}: {e}")
                continue
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Get top_k results and add title information
        results = []
        for item in similarities[:top_k]:
            filename = item['filename']
            
            # Extract image key from filename (remove extension)
            image_key = os.path.splitext(filename)[0]
            
            # Look up title information
            title_info = self.image_titles.get(image_key, {})
            
            result = {
                'filename': filename,
                'similarity': item['similarity'],
                'confidence': item['similarity'],  # Use similarity as confidence
                'title': title_info.get('title', 'Unknown'),
                'cultural_number': title_info.get('cultural_number', 'N/A'),
                'cultural_category': title_info.get('cultural_category', 'Unknown'),
                'cultural_dynasty': title_info.get('cultural_dynasty', 'Unknown'),
                'image_path': os.path.join(self.image_folder, filename)
            }
            
            results.append(result)
        
        return results
    
    def get_database_stats(self):
        """Get statistics about the feature database"""
        stats = {
            'total_images': len(self.feature_database),
            'feature_model': self.feature_model_name,
            'has_pca': self.pca is not None,
            'feature_dimension': None,
            'database_file_exists': os.path.exists(self.feature_db_file)
        }
        
        if self.feature_database:
            first_feature = next(iter(self.feature_database.values()))
            stats['feature_dimension'] = len(first_feature) if hasattr(first_feature, '__len__') else 'Unknown'
        
        return stats

def main():
    """Test the similarity detector"""
    print("🎨 Cultural Artifact Similarity Detector")
    print("=" * 50)
    
    # Initialize detector
    detector = SimilarityDetector(feature_model='VGG16')
    
    # Build feature database
    detector.build_feature_database()
    
    # Show statistics
    stats = detector.get_database_stats()
    print("\n📊 Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Similarity detector ready!")
    print("Use the main UI to test similarity detection.")

if __name__ == "__main__":
    main()
