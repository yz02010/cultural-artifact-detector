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

## 🏛️ Supported Categories

The system can identify **12 types** of cultural artifacts:

1. **Bronze, Brass, and Copper** - Ancient metalwork and vessels
2. **Carvings** - Stone, wood, and ivory sculptures  
3. **Ceramics** - Pottery, porcelain, and glazed items
4. **Enamels** - Decorative enamelware and cloisonné
5. **Gold and Silver** - Precious metal artifacts and jewelry
6. **Imperial Seals and Albums** - Official seals and documents
7. **Lacquer** - Traditional lacquerware and furniture
8. **Other Crafts** - Miscellaneous traditional crafts
9. **Paintings** - Traditional scrolls and artwork
10. **Sculpture** - Three-dimensional art pieces
11. **Textiles** - Fabrics, clothing, and tapestries
12. **Timepieces and Instruments** - Clocks, scientific instruments

## 🎯 Features

- **🤖 AI Classification** - TensorFlow MobileNetV2 with transfer learning
- **🖼️ Background Removal** - Automatic gray/black background detection
- **🖱️ Interactive Selection** - Draw rectangles on screen to capture objects
- **📊 Confidence Scores** - Get prediction confidence for all categories
- **📁 Batch Processing** - Process multiple images with CSV export
- **💾 Result Export** - Save predictions and visualizations
- **🎨 Modern UI** - Intuitive graphical interface

## 📋 Requirements

```
tensorflow>=2.10.0
opencv-python>=4.6.0
Pillow>=9.2.0
numpy>=1.21.0
pyautogui>=0.9.54
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Documentation

- **[HOW_TO_USE_H5_MODEL.md](HOW_TO_USE_H5_MODEL.md)** - Complete usage guide
- **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - Project overview and performance metrics

## 🏗️ Project Structure

```
├── main_ui.py                     # Main UI application (NEW)
├── screen_capture_classifier.py   # Interactive screen capture
├── cultural_detector_improved.h5  # Trained model (30.3 MB)
├── test_h5_model.py               # Model testing and validation
├── predict.py                     # Prediction utilities
├── image_preprocessor.py          # Background removal tools
├── improved_trainer.py            # Model training script
├── demo.py                        # Dataset analysis and demo
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 📊 Model Performance

- **Training Accuracy**: 66.3%
- **Validation Accuracy**: 81.3%
- **Architecture**: MobileNetV2 with Transfer Learning
- **Dataset Size**: 316 cultural artifact images
- **Model Size**: 30.3 MB

## 🎮 Usage Examples

### Screen Capture Detection
1. Run `python main_ui.py`
2. Click "Screen Capture Detection"
3. Draw a rectangle around any artifact on your screen
4. Get instant AI classification results!

### File Upload
1. Click "Upload Image File" in the main UI
2. Select an image containing a cultural artifact
3. View classification results with confidence scores

### Batch Processing
1. Click "Batch Process Folder"
2. Select a folder containing artifact images
3. Export results to CSV for further analysis

## 🔧 Advanced Usage

### Python Integration
```python
from predict import CulturalArtifactPredictor

# Initialize predictor
predictor = CulturalArtifactPredictor()

# Classify single image
results = predictor.predict('artifact.jpg', top_k=3)

# Process batch
batch_results = predictor.predict_batch('image_folder/', top_k=3)
```

### Custom Preprocessing
```python
from image_preprocessor import ImagePreprocessor

processor = ImagePreprocessor()
clean_image = processor.remove_gray_black_background(image)
enhanced_image = processor.enhance_contrast(clean_image)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the excellent deep learning framework
- MobileNetV2 architecture for efficient transfer learning
- Cultural institutions for providing artifact datasets

---

**Ready to explore cultural artifacts with AI? Start with `python main_ui.py`!** 🎨
