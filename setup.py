"""
Setup script for Cultural Artifact Detection System
This script installs all required dependencies and sets up the environment
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}")
        return False

def check_package(package_name):
    """Check if a package is already installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    """Main setup function"""
    print("Cultural Artifact Detection System - Setup")
    print("=" * 50)
    
    # List of required packages
    packages = [
        "tensorflow>=2.15.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0"
    ]
    
    # Optional packages for advanced features
    optional_packages = [
        "albumentations>=1.3.0",
        "seaborn>=0.12.0",
        "rembg>=2.0.50"
    ]
    
    print("Installing core packages...")
    failed_packages = []
    
    for package in packages:
        package_name = package.split(">=")[0].split("==")[0]
        print(f"\nInstalling {package_name}...")
        
        if not install_package(package):
            failed_packages.append(package)
    
    print("\nInstalling optional packages...")
    optional_failed = []
    
    for package in optional_packages:
        package_name = package.split(">=")[0].split("==")[0]
        print(f"\nInstalling {package_name} (optional)...")
        
        if not install_package(package):
            optional_failed.append(package)
    
    # Summary
    print("\n" + "=" * 50)
    print("INSTALLATION SUMMARY")
    print("=" * 50)
    
    if not failed_packages:
        print("✓ All core packages installed successfully!")
    else:
        print("✗ Some core packages failed to install:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
    
    if optional_failed:
        print("\n⚠ Some optional packages failed to install:")
        for pkg in optional_failed:
            print(f"  - {pkg}")
        print("  (These are not required for basic functionality)")
    
    # Test imports
    print("\nTesting package imports...")
    test_packages = {
        'numpy': 'numpy',
        'cv2': 'opencv-python',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'PIL': 'pillow',
        'tensorflow': 'tensorflow'
    }
    
    working_packages = []
    broken_packages = []
    
    for import_name, package_name in test_packages.items():
        try:
            __import__(import_name)
            working_packages.append(package_name)
            print(f"✓ {package_name}")
        except ImportError:
            broken_packages.append(package_name)
            print(f"✗ {package_name}")
    
    print("\n" + "=" * 50)
    if not broken_packages:
        print("🎉 Setup completed successfully!")
        print("You can now run the training scripts:")
        print("  python simple_trainer.py")
        print("  python cultural_artifact_detector.py")
    else:
        print("⚠ Setup completed with some issues.")
        print("Please manually install the failed packages:")
        for pkg in broken_packages:
            print(f"  pip install {pkg}")
    
    print("\nNext steps:")
    print("1. Ensure your images are in the 'image_test' directory")
    print("2. Ensure your mapping file is at 'get_data/image_title_mapping.json'")
    print("3. Run the preprocessing script: python image_preprocessor.py")
    print("4. Run the training script: python simple_trainer.py")

if __name__ == "__main__":
    main()
