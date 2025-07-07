"""
Build Feature Database for Similarity Detection
==============================================

This script pre-builds the feature database for the similarity detection system.
Run this script once to extract features from all images in the image library.

Usage:
    python build_feature_database.py
"""

import os
import sys
from similarity_detector import SimilarityDetector

def main():
    print("🎨 Cultural Artifact Similarity Detection")
    print("Building Feature Database")
    print("=" * 50)
    
    # Check if image folder exists
    if not os.path.exists('image'):
        print("❌ Error: 'image' folder not found!")
        print("Please make sure the image folder contains your artifact images.")
        return
    
    # Check if mapping file exists
    if not os.path.exists('get_data/image_title_mapping.json'):
        print("❌ Error: 'get_data/image_title_mapping.json' not found!")
        print("Please make sure the mapping file exists.")
        return
    
    try:
        # Initialize similarity detector
        print("🔄 Initializing similarity detector...")
        detector = SimilarityDetector(feature_model='VGG16')
        
        # Ask user if they want to rebuild existing database
        if os.path.exists(detector.feature_db_file):
            response = input(f"\n📦 Feature database already exists: {detector.feature_db_file}\n"
                           "Do you want to rebuild it? (y/N): ").strip().lower()
            force_rebuild = response in ['y', 'yes']
        else:
            force_rebuild = True
        
        # Build feature database
        print(f"🔄 Building feature database (force_rebuild={force_rebuild})...")
        detector.build_feature_database(force_rebuild=force_rebuild)
        
        # Show statistics
        stats = detector.get_database_stats()
        print("\n📊 Database Statistics:")
        print("-" * 30)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print(f"\n✅ Feature database ready!")
        print(f"💾 Database saved as: {detector.feature_db_file}")
        print(f"🎯 You can now use similarity detection in the main UI!")
        
    except Exception as e:
        print(f"❌ Error building feature database: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
