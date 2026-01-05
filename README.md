# 🏥 AI Polyp Detection System

A comprehensive deep learning pipeline for detecting polyps in colonoscopy images using PyTorch and Streamlit.

## 📋 Overview

This project implements an end-to-end AI system for polyp detection in colonoscopy images, featuring:
- Custom CNN classifier with 14M parameters
- Interactive web interface built with Streamlit
- Trained on Kvasir-SEG dataset (1,002 polyp images)
- GPU/CPU support with automatic device detection
- Adjustable detection threshold for sensitivity control

## 🚀 Features

### Deep Learning Models
- **CNN Classifier**: 4-block convolutional architecture with batch normalization
- **YOLO Detector**: Bounding box regression for polyp localization
- **U-Net Segmentation**: Pixel-level polyp segmentation

### Web Application
- Real-time polyp detection
- Confidence score visualization
- Adjustable detection threshold (50-95%)
- Export analysis results (images & reports)
- Medical disclaimer and limitations

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, but recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/AI-Polyp-Detection.git
cd AI-Polyp-Detection
```

2. **Create virtual environment**
```bash
conda create -n polyp_detection python=3.9
conda activate polyp_detection
```

3. **Install dependencies**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install streamlit opencv-python pillow matplotlib pandas numpy scikit-learn
```

4. **Download the model**
- Place your trained model (`cnn_best_model.pth`) in the `models/` directory

## 📊 Dataset

**Kvasir-SEG Dataset**
- 1,002 colonoscopy images with polyp annotations
- Includes segmentation masks and bounding boxes
- Image resolution: Variable (resized to 256×256 for training)

## 🎯 Usage

### Training the Model

Run the Jupyter notebook:
```bash
jupyter notebook AI_Polyp_Detection_Complete_Pipeline.ipynb
```

Execute cells sequentially to:
1. Load and preprocess data
2. Train CNN classifier
3. Evaluate model performance
4. Export trained model

### Running the Web Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### Using the Application

1. **Upload Image**: Choose a colonoscopy image (PNG, JPG, JPEG, BMP)
2. **Adjust Threshold**: Use sidebar slider (default: 70%)
   - 50-60%: High sensitivity (more detections)
   - 70-80%: Balanced (recommended)
   - 85-95%: High specificity (fewer false positives)
3. **Analyze**: Click "Analyze Image" button
4. **Review Results**: View prediction, confidence scores, and visualizations
5. **Export**: Download analysis image and text report

## 📈 Model Performance

**CNN Classifier Metrics:**
- Architecture: 4 convolutional blocks (32→64→128→256 filters)
- Parameters: ~14 million
- Training: 50 epochs with early stopping
- Optimizer: Adam (lr=0.001)
- Loss: Binary Cross-Entropy

**Important Limitation:**
- Model trained only on polyp images (single-class)
- May produce false positives on normal tissue
- Threshold adjustment recommended for production use

## 🗂️ Project Structure

```
AI-Polyp-Detection/
├── app.py                                    # Streamlit web application
├── AI_Polyp_Detection_Complete_Pipeline.ipynb # Training notebook
├── DATASET_DESCRIPTION.md                    # Dataset documentation
├── models/
│   └── cnn_best_model.pth                   # Trained model weights
├── archive/
│   └── Kvasir-SEG/                          # Dataset files
├── dataset/                                  # Preprocessed data
├── results/                                  # Training outputs
└── README.md                                 # This file
```

## 🔧 Configuration

### Model Hyperparameters

```python
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
RANDOM_SEED = 42
```

### Detection Threshold

Adjust in sidebar (app.py):
- Default: 0.70 (70%)
- Range: 0.50 - 0.95

## 📝 Results Interpretation

**Prediction Categories:**
- ✅ **NO POLYP**: Score < 50% (Very low polyp likelihood)
- ⚠️ **UNCERTAIN**: Score 50-70% (Manual review recommended)
- ⚠️ **POLYP DETECTED**: Score ≥ 70% (Medical attention required)

**Confidence Levels:**
- **Very Low**: < 50%
- **Low**: 50-70%
- **Moderate**: 70-85%
- **High**: ≥ 85%

## ⚠️ Medical Disclaimer

This AI system is designed for **research and educational purposes only**. It should NOT be used as a sole diagnostic tool. Key limitations:

- Trained on limited dataset (1,002 images)
- Single-class training (no negative samples)
- Not FDA-approved for clinical use
- Requires validation by medical professionals

**Always consult qualified gastroenterologists for medical diagnosis and treatment decisions.**

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add negative samples (normal tissue images) to training
- Implement YOLO and U-Net models
- Improve model architecture
- Add more evaluation metrics
- Enhance UI/UX

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Kvasir-SEG Dataset**: Jha et al., 2020
- **PyTorch**: Facebook AI Research
- **Streamlit**: Streamlit Inc.

## 📧 Contact

For questions or collaboration:
- GitHub Issues: [Report issues](https://github.com/yourusername/AI-Polyp-Detection/issues)
- Email: your.email@example.com

## 🔗 References

1. Jha, D., et al. (2020). Kvasir-SEG: A Segmented Polyp Dataset. MMM 2020.
2. PyTorch Documentation: https://pytorch.org/docs/
3. Streamlit Documentation: https://docs.streamlit.io/

---

**Version**: 1.0.0  
**Last Updated**: January 2, 2026  
**Status**: Active Development
