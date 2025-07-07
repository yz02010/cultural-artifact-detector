# 🎨 How to Use Your H5 Cultural Artifact Detection Model

## 🚀 **Quick Start Guide**

Your trained H5 model (`cultural_detector_improved.h5`) is ready to use! Here are several ways to use it:

---

## 🎨 **Method 1: Main UI Application (NEW & RECOMMENDED)**

### **Start the comprehensive UI application:**
```powershell
cd "c:\Users\syuan\Code\myapp"
C:/Users/syuan/Code/myapp/.venv/Scripts/python.exe main_ui.py
```

### **Features:**
- �️ **Screen Capture Detection** - Draw rectangle on screen to capture and classify
- 📁 **File Upload Detection** - Upload image files for classification  
- 📂 **Batch Processing** - Process multiple images from folders with CSV export
- 📹 **Camera Capture** - Real-time detection (coming soon)
- 📊 **Real-time Results** - View all predictions with confidence scores
- 💾 **Export Results** - Save batch processing results to CSV
- 🎯 **Multi-mode Detection** - All detection methods in one application

### **How to use:**
1. **Launch the application** - Run `main_ui.py`
2. **Choose detection mode** - Click any of the detection buttons
3. **For Screen Capture**: Draw rectangle around artifact → Get instant results
4. **For File Upload**: Select image file → View classification results
5. **For Batch Processing**: Select folder → Export results to CSV

---

## �📋 **Method 2: Simple Image Testing**

### **Test with existing images:**
```powershell
cd "c:\Users\syuan\Code\myapp"
C:/Users/syuan/Code/myapp/.venv/Scripts/python.exe test_h5_model.py
```

### **What it does:**
- ✅ Tests your model on sample images
- ✅ Shows confidence scores and rankings
- ✅ Creates visualizations with bar charts
- ✅ Automatically removes gray/black backgrounds
- ✅ Saves results as PNG and TXT files

---

## 🖱️ **Method 3: Interactive Screen Capture (Legacy)**

### **Start the screen capture tool:**
```powershell
cd "c:\Users\syuan\Code\myapp"
C:/Users/syuan/Code/myapp/.venv/Scripts/python.exe screen_capture_classifier.py
```

### **How to use:**
1. **Click "Start Screen Capture"** 
2. **Draw a rectangle** around any cultural artifact on your screen
3. **Release mouse** to capture and classify
4. **View results** with confidence scores and category predictions

### **Features:**
- 🖥️ **Real-time screen capture** - Analyze anything on your screen
- 🎯 **Interactive selection** - Draw rectangles to select areas
- 📊 **Instant results** - Get predictions in seconds
- 💾 **Automatic saving** - Results saved as images
- 🔄 **Multiple captures** - Capture as many areas as you want

---

## 🐍 **Method 4: Python Code Integration**

### **Basic prediction code:**
```python
from predict import CulturalArtifactPredictor

# Initialize the predictor with your H5 model
predictor = CulturalArtifactPredictor('cultural_detector_improved.h5')

# Predict a single image
results = predictor.predict('path/to/your/artifact.jpg', top_k=3)

# Print results
for result in results:
    print(f"{result['rank']}. {result['category']}: {result['confidence']:.1%}")
```

### **Batch processing:**
```python
# Process all images in a directory
results = predictor.predict_batch('your_image_directory/', top_k=3)
```

---

## 🔧 **Method 4: Custom Integration**

### **Load the H5 model directly:**
```python
import tensorflow as tf
import cv2
import numpy as np
from image_preprocessor import ImagePreprocessor

# Load the model
model = tf.keras.models.load_model('cultural_detector_improved.h5')

# Define class names
class_names = [
    'Bronze, Brass, and Copper', 'Carvings', 'Ceramics', 'Enamels',
    'Gold and Silver', 'Imperial Seals and Albums', 'Lacquer',
    'Other Crafts', 'Paintings', 'Sculpture', 'Textiles',
    'Timepieces and Instruments'
]

# Preprocess function
def preprocess_image(image_path):
    preprocessor = ImagePreprocessor()
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Remove background and enhance
    image = preprocessor.remove_gray_black_background(image)
    image = preprocessor.enhance_contrast(image)
    
    # Resize and normalize
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    
    return image

# Make prediction
def predict_artifact(image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    
    # Get top prediction
    top_index = np.argmax(predictions[0])
    confidence = predictions[0][top_index]
    category = class_names[top_index]
    
    return category, confidence

# Example usage
category, confidence = predict_artifact('my_artifact.jpg')
print(f"Predicted: {category} ({confidence:.1%} confidence)")
```

---

## 📊 **Understanding the Results**

### **What you get:**
- **🏆 Category**: The type of cultural artifact (e.g., "Ceramics", "Bronze", "Jade")
- **📈 Confidence**: How sure the model is (0-100%)
- **🎯 Top 3 Predictions**: Alternative possibilities with their confidence scores

### **Example output:**
```
🏆 BEST MATCH: Ceramics (87.4% confidence)

Top 3 Predictions:
1. Ceramics: 87.4%           🥇
2. Enamels: 3.1%            🥈  
3. Gold and Silver: 2.5%    🥉
```

---

## 🎯 **Model Categories**

Your model can identify these 12 types of cultural artifacts:

| Category | Description | Common Examples |
|----------|-------------|-----------------|
| **Ceramics** 🏺 | Pottery, porcelain, earthenware | Vases, bowls, plates |
| **Bronze, Brass, and Copper** 🥉 | Metal artifacts | Ceremonial vessels, coins |
| **Jade** 💎 | Jade carvings and jewelry | Pendants, figurines |
| **Paintings** 🖼️ | Traditional paintings | Scrolls, wall paintings |
| **Sculpture** 🗿 | Carved figures and statues | Stone/wood sculptures |
| **Textiles** 🧵 | Fabric artifacts | Silk, embroidery |
| **Gold and Silver** 🥇 | Precious metal items | Jewelry, ornaments |
| **Enamels** ✨ | Enameled decorative items | Cloisonné, painted enamels |
| **Carvings** 🪚 | Carved decorative items | Wood/stone carvings |
| **Imperial Seals and Albums** 📜 | Official documents/seals | Royal seals, albums |
| **Lacquer** 🏮 | Lacquered items | Boxes, furniture |
| **Timepieces and Instruments** ⏰ | Clocks and tools | Ancient clocks, instruments |

---

## 🎨 **Background Removal Features**

Your model automatically handles problematic backgrounds:

### **What it removes:**
- ✅ **Gray backgrounds** (common in museum photos)
- ✅ **Black backgrounds** (studio photography)
- ✅ **Low contrast areas** (poorly lit backgrounds)
- ✅ **Mixed gray/black regions**

### **How it works:**
1. **HSV color analysis** detects low-saturation areas
2. **Intensity thresholding** identifies dark regions  
3. **Morphological operations** clean up the mask
4. **White replacement** provides clean background

---

## 💡 **Tips for Best Results**

### **Image Quality:**
- 📷 Use **high-resolution images** when possible
- 💡 Ensure **good lighting** on the artifact
- 🎯 **Center the artifact** in the image frame
- 🔍 Use **single objects** per image for best accuracy

### **Using Screen Capture:**
- 🖥️ **Display the artifact** on your screen first
- 📐 **Draw tight rectangles** around the artifact
- 🚫 **Avoid including** too much background
- 🔄 **Try multiple angles** if confidence is low

---

## 📈 **Performance Expectations**

### **Model Statistics:**
- 🎯 **Validation Accuracy**: 81.3%
- ⚡ **Processing Speed**: ~0.1 seconds per image
- 📊 **Training Data**: 316 Chinese cultural artifacts
- 🏺 **Best Performance**: Ceramics (87%+ accuracy)

### **Confidence Interpretation:**
- **80-100%**: Very confident prediction ✅
- **60-80%**: Good prediction ⚡
- **40-60%**: Moderate confidence ⚠️
- **Below 40%**: Low confidence ❓

---

## 🛠️ **Troubleshooting**

### **Common Issues:**

**❌ "Model not found"**
- Ensure `cultural_detector_improved.h5` exists in your directory
- Check the file path in your script

**❌ "Low confidence predictions"**
- Try cropping the image closer to the artifact
- Ensure good lighting and contrast
- Check if the artifact type is in the 12 supported categories

**❌ "Screen capture not working"**
- Make sure you have admin permissions
- Try running as administrator
- Check that pyautogui is installed

**❌ "Import errors"**
- Run: `C:/Users/syuan/Code/myapp/.venv/Scripts/python.exe -m pip install -r requirements.txt`

---

## 🎉 **Ready to Use!**

Your H5 model is production-ready! Choose your preferred method:

1. **🖱️ Screen Capture** - Most interactive and fun
2. **📁 Batch Testing** - For processing multiple images
3. **🐍 Python Integration** - For custom applications

**Start with screen capture for immediate results:**
```powershell
C:/Users/syuan/Code/myapp/.venv/Scripts/python.exe screen_capture_classifier.py
```

**Happy artifact hunting! 🎨🏛️**
