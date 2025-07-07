# 🎨 Cultural Artifact Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-powered system for detecting and classifying cultural artifacts using TensorFlow and computer vision. Features automatic background removal and multiple detection modes including interactive screen capture.

![Demo Screenshot](sample_images_demo.png)

## 🚀 Quick Start

### **Run the Main Application**
```bash
python main_ui.py
```

This opens a comprehensive GUI with multiple detection modes:
- 🖥️ **Screen Capture** - Draw rectangle on screen to capture objects
- 📁 **File Upload** - Upload and classify image files
- 📂 **Batch Processing** - Process folders with CSV export
- 📹 **Camera Capture** - Real-time detection (coming soon)

### **Alternative: Screen Capture Only**
```bash
python screen_capture_classifier.py
```
├── setup.py                      # Package installation script
├── requirements.txt              # Python dependencies
├── image_test/                   # Your training images
├── get_data/
│   └── image_title_mapping.json  # Image metadata
└── processed_images/             # Output directory for processed images
```

## Quick Start

### 1. Install Dependencies

All required packages have been installed. If you need to reinstall:

```powershell
C:/Users/syuan/Code/myapp/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 2. Run the Demo

To see how the system works with your data:

```powershell
C:/Users/syuan/Code/myapp/.venv/Scripts/python.exe demo.py
```

This will:
- Analyze your dataset structure
- Show sample images with preprocessing
- Demonstrate background removal
- Create visualizations

### 3. Preprocess Images

To preprocess all images and remove backgrounds:

```powershell
C:/Users/syuan/Code/myapp/.venv/Scripts/python.exe image_preprocessor.py
```

### 4. Train the Model

For quick training with a simple model:

```powershell
C:/Users/syuan/Code/myapp/.venv/Scripts/python.exe simple_trainer.py
```

For advanced training with more features:

```powershell
C:/Users/syuan/Code/myapp/.venv/Scripts/python.exe cultural_artifact_detector.py
```

## Background Removal

The system automatically handles gray and black backgrounds using several techniques:

1. **Color-based Masking**: Identifies gray and black areas using HSV color space
2. **Saturation Analysis**: Detects low-saturation (grayish) regions
3. **Morphological Operations**: Cleans up the detected background mask
4. **White Replacement**: Replaces background with clean white color

### Parameters you can adjust:

In `image_preprocessor.py`, you can modify:
- `threshold_low`: Lower threshold for dark areas (default: 30)
- `threshold_high`: Upper threshold for gray areas (default: 80)

## Data Format

Your images should be organized as:
- Images in `image_test/` directory
- Metadata in `get_data/image_title_mapping.json`

The mapping file should have this structure:
```json
{
  "image_1_0": {
    "title": "Artifact Name",
    "cultural_number": "ID123",
    "cultural_category": "Category Name",
    "cultural_dynasty": "Dynasty Name"
  }
}
```

## Model Architecture

### Simple Model (simple_trainer.py)
- Basic CNN with 4 convolutional layers
- Global average pooling
- Dropout for regularization
- Suitable for quick experiments

### Advanced Model (cultural_artifact_detector.py)
- EfficientNetB0 backbone (pre-trained on ImageNet)
- Transfer learning with fine-tuning
- Advanced data augmentation
- Multiple metrics tracking

## Training Results

After training, you'll get:
- Trained model saved as `.h5` file
- Training history plots
- Classification accuracy metrics
- Top-3 accuracy for multi-class evaluation

## Making Predictions

```python
from simple_trainer import SimpleCulturalDetector

# Load trained model
detector = SimpleCulturalDetector()
detector.model = tf.keras.models.load_model('simple_cultural_detector.h5')

# Predict on new image
category, confidence = detector.predict('path/to/new/image.jpg')
print(f"Predicted category: {category} (confidence: {confidence:.3f})")
```

## Customization

### Adjusting Background Removal
Modify the thresholds in `ImagePreprocessor.remove_gray_black_background()`:
- Increase `threshold_low` to remove more dark areas
- Increase `threshold_high` to remove more gray areas

### Changing Image Size
Modify `img_size` parameter in the detector classes:
```python
detector = SimpleCulturalDetector()
detector.img_size = (299, 299)  # For larger images
```

### Adding More Augmentation
In `cultural_artifact_detector.py`, modify the `transform` pipeline to add more augmentation techniques.

## Troubleshooting

### Common Issues:

1. **"No data found"**: Check that your images are in `image_test/` and mapping file exists
2. **Memory errors**: Reduce batch size or image size
3. **Poor accuracy**: Try more training epochs or data augmentation
4. **Background not removed**: Adjust threshold parameters in `remove_gray_black_background()`

### Performance Tips:

1. **More training data**: Collect more diverse images for better accuracy
2. **Data balancing**: Ensure balanced representation across categories
3. **Transfer learning**: Use pre-trained models for better performance
4. **Hyperparameter tuning**: Experiment with learning rates and architecture

## Next Steps

1. Run the demo to see how your data looks
2. Experiment with background removal parameters
3. Train the simple model first to verify everything works
4. Scale up to the advanced model for better accuracy
5. Evaluate results and iterate on the preprocessing

## Support

The system is designed to work with your cultural artifact images. The background removal specifically targets gray and black backgrounds common in museum photography.

For best results:
- Ensure good image quality
- Have balanced data across categories
- Experiment with preprocessing parameters
- Use transfer learning for better performance
