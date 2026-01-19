# 🏥 AI-Based Polyp Detection System

A comprehensive AI-powered system for detecting and classifying polyps in colonoscopy images using **YOLO deep learning models** with a modern **React + TypeScript frontend** and **FastAPI backend**.

## 📋 Overview

This project implements an end-to-end clinical-grade polyp detection system featuring:
- **YOLO Multi-Class Detection**: Real-time polyp detection with 4 classes (Normal, Hyperplastic, Adenomatous, Adenocarcinoma)
- **Modern React Frontend**: Clinical View Pro - Professional medical interface with TypeScript
- **FastAPI Backend**: High-performance REST API with YOLO integration
- **Risk Assessment**: Automatic risk level calculation (Low, Medium, High, Critical)
- **PDF Export**: Generate clinical reports for documentation
- **GPU/CPU Support**: Automatic device detection for optimal performance

## 🚀 Features

### 🎯 AI Detection System
- **YOLO11/YOLO8 Models**: State-of-the-art object detection
- **Multi-Class Classification**: 
  - Normal tissue
  - Hyperplastic Polyp (Low risk)
  - Adenomatous Polyp (Medium-High risk)
  - Adenocarcinoma (Critical risk)
- **Bounding Box Detection**: Precise polyp localization
- **Confidence Scoring**: Detailed confidence metrics (0-100%)

### 🖥️ Frontend (React + TypeScript)
- **Modern UI**: Built with React, Vite, TailwindCSS, and shadcn/ui
- **Real-time Analysis**: Instant polyp detection with visual feedback
- **Side-by-Side Comparison**: Original vs Annotated images
- **Risk Dashboard**: Color-coded risk levels with recommendations
- **Export Reports**: Generate and download PDF reports
- **Responsive Design**: Mobile-friendly interface

### ⚡ Backend (Python FastAPI)
- **RESTful API**: Fast, async API with automatic documentation
- **YOLO Integration**: Ultralytics YOLO models (YOLOv8/YOLO11)
- **Image Processing**: PIL, OpenCV, NumPy for image handling
- **CORS Support**: Configured for frontend communication
- **Swagger Docs**: Auto-generated API documentation at `/docs`

## 🛠️ Installation

### Prerequisites
- **Python 3.8+** with pip
- **Node.js 18+** with npm
- **CUDA-capable GPU** (optional, recommended for faster inference)
- **Conda** (recommended for environment management)

### Setup Instructions

#### 1. Clone the Repository
```bash
git clone https://github.com/S-PANDIYAN/AI-Based-Polyp-Detection-System.git
cd AI-Based-Polyp-Detection-System
```

#### 2. Setup Python Environment
```bash
# Create conda environment
conda create -n polyp_detection python=3.9
conda activate polyp_detection

# Install Python dependencies
cd clinical-view-frontend
pip install -r requirements.txt
```

#### 3. Install Node.js Dependencies
```bash
# In the clinical-view-frontend directory
npm install
```

#### 4. Configure Environment
Create a `.env` file in `clinical-view-frontend/`:
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_MOCK_API=false
```

## 🚀 Running the Application

### Option 1: Quick Start (Windows)
```bash
cd clinical-view-frontend
start_all.bat
```
This automatically starts both backend and frontend servers.

### Option 2: Manual Start

**Terminal 1 - Start Backend API:**
```bash
cd clinical-view-frontend
conda activate polyp_detection
python yolo_api_server.py
```
Backend runs at: **http://localhost:8000**

**Terminal 2 - Start Frontend:**
```bash
cd clinical-view-frontend
npm run dev
```
Frontend runs at: **http://localhost:5173**

### Access Points
- **Frontend UI**: http://localhost:5173
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## 📊 Dataset & Models

### Kvasir-SEG Dataset
- **1,002 colonoscopy images** with polyp annotations
- Includes segmentation masks and bounding boxes
- Image resolution: Variable (preprocessed to 640×640 for YOLO)

### YOLO Models

**Available Models:**
1. **yolo11n.pt** (5.35 MB) - Base YOLO11 nano model
2. **yolov8n.pt** (6.25 MB) - Base YOLO8 nano model
3. **best.pt** (5.94 MB) - **Trained multi-class polyp detector** ⭐
   - Location: `yolo_clinical_training/polyp_multiclass/weights/best.pt`
   - Classes: Normal, Hyperplastic, Adenomatous, Adenocarcinoma

### Additional Models (Local Only - Not in Repository)
- CNN Classifier (133 MB)
- U-Net Segmentation (118 MB)

## 🎯 Usage

### 1. Upload Colonoscopy Image
- Supported formats: PNG, JPG, JPEG, BMP
- Recommended resolution: 640×640 or higher

### 2. Analyze Image
- Click **"Analyze Image"** or **"Drop new scan"**
- Wait for YOLO inference (typically < 1 second)

### 3. Review Results
- **Detection Type**: Polyp class (Normal/Hyperplastic/Adenomatous/Adenocarcinoma)
- **Confidence Score**: Percentage (0-100%)
- **Risk Level**: Color-coded (Low/Medium/High/Critical)
- **Bounding Boxes**: Visual polyp localization
- **Recommendations**: Clinical guidance based on detection

### 4. Export Report
- **Download Report**: PDF with analysis summary
- **Export Mask**: PNG with highlighted annotations

## 📈 Model Performance

**YOLO Multi-Class Detector:**
- **Model**: YOLOv8n / YOLO11n fine-tuned
- **Classes**: 4 (Normal, Hyperplastic, Adenomatous, Adenocarcinoma)
- **Input Size**: 640×640
- **Inference Time**: < 100ms on GPU, < 500ms on CPU
- **Confidence Threshold**: Adjustable (default: 0.25)

**Risk Classification:**
- ✅ **Normal**: No polyp detected → **Low Risk**
- ⚠️ **Hyperplastic Polyp**: Low malignant potential → **Low-Medium Risk**
- 🟡 **Adenomatous Polyp**: Pre-cancerous lesion → **Medium-High Risk**
- 🔴 **Adenocarcinoma**: Malignant cancer → **Critical Risk**

**Performance Metrics:**
- Precision: High accuracy in polyp localization
- Recall: Effective detection of various polyp types
- F1-Score: Balanced precision and recall
- mAP@0.5: Optimized for medical imaging

## 🗂️ Project Structure

```
AI-Based-Polyp-Detection-System/
├── clinical-view-frontend/           # Main Application
│   ├── yolo_api_server.py           # FastAPI Backend ⭐
│   ├── requirements.txt             # Python dependencies
│   ├── package.json                 # Node.js dependencies
│   ├── start_all.bat                # Windows startup script
│   ├── src/
│   │   ├── components/              # React components
│   │   ├── pages/                   # Page components
│   │   ├── services/                # API services
│   │   └── lib/                     # Utilities
│   └── public/                      # Static assets
│
├── yolo_clinical_training/          # YOLO Training
│   └── polyp_multiclass/
│       └── weights/
│           └── best.pt              # Trained YOLO model ⭐
│
├── models/                          # Additional models (local)
│   ├── cnn_best_model.pth          # CNN classifier
│   └── unet_segmentation_best.pth  # U-Net segmentation
│
├── UI/                              # Static HTML mockups
│   ├── ai_polyp_detection_system_dashboard_1/
│   ├── ai_polyp_detection_system_dashboard_2/
│   └── ...                          # Various UI prototypes
│
├── dataset/                         # Dataset (excluded from git)
├── archive/                         # Kvasir-SEG (excluded from git)
├── yolo_dataset/                    # YOLO training data (excluded)
│
├── AI_Polyp_Detection_Complete_Pipeline.ipynb  # Training notebook
├── Clinical_Polyp_Classification_Segmentation.ipynb
├── yolo11n.pt                       # Base YOLO11 model
├── yolov8n.pt                       # Base YOLO8 model
├── README.md                        # This file
└── GITHUB_UPLOAD_COMMANDS.md       # Git deployment guide
```

## 🔧 API Documentation

### Backend Endpoints

**Base URL:** `http://localhost:8000`

#### 1. Health Check
```http
GET /api/health
```
Returns API status and model information.

#### 2. Polyp Detection
```http
POST /api/detect
Content-Type: multipart/form-data

Body:
  - file: image file (PNG, JPG, JPEG, BMP)
  - confidence: float (optional, default: 0.25)
```

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "class": "Adenomatous Polyp",
      "confidence": 0.952,
      "bbox": [120, 85, 340, 280],
      "risk_level": "High"
    }
  ],
  "annotated_image": "base64_encoded_image",
  "processing_time": 0.087
}
```

### Frontend API Service

Located at: `clinical-view-frontend/src/services/polypDetectionAPI.ts`

**Methods:**
- `analyzeImage(file, confidence)` - Upload and analyze image
- `getHealth()` - Check API health status

## 🔧 Configuration

### Backend Configuration (`yolo_api_server.py`)

```python
MODEL_PATH = "../yolo_clinical_training/polyp_multiclass/weights/best.pt"

CLASS_NAMES = {
    0: "Normal",
    1: "Hyperplastic Polyp",
    2: "Adenomatous Polyp",
    3: "Adenocarcinoma"
}

RISK_LEVELS = {
    "Normal": "Low",
    "Hyperplastic Polyp": "Low",
    "Adenomatous Polyp": "High",
    "Adenocarcinoma": "Critical"
}
```

### Frontend Configuration (`.env`)

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_MOCK_API=false
```

## 📝 Results Interpretation

### Detection Categories

| Class | Description | Risk Level | Color Code | Action Required |
|-------|-------------|------------|------------|-----------------|
| **Normal** | No polyp detected | Low | 🟢 Green | Routine monitoring |
| **Hyperplastic Polyp** | Low malignant potential | Low-Medium | 🟡 Yellow | Regular surveillance |
| **Adenomatous Polyp** | Pre-cancerous lesion | Medium-High | 🟠 Orange | Removal recommended |
| **Adenocarcinoma** | Malignant cancer | Critical | 🔴 Red | Immediate intervention |

### Confidence Levels

- **< 50%**: Very Low confidence - May require re-scan
- **50-70%**: Low confidence - Manual review recommended
- **70-85%**: Moderate confidence - Clinical correlation advised
- **85-95%**: High confidence - Reliable detection
- **> 95%**: Very High confidence - Strong detection

### Clinical Recommendations

**For Each Detection Class:**

✅ **Normal**
- Continue routine screening schedule
- No immediate action required
- Follow standard colonoscopy guidelines

⚠️ **Hyperplastic Polyp**
- Generally benign, low risk
- Consider surveillance colonoscopy
- Monitor for size changes

🟡 **Adenomatous Polyp**
- Pre-cancerous potential
- Polypectomy recommended
- Follow-up surveillance per guidelines

🔴 **Adenocarcinoma**
- Immediate medical attention required
- Biopsy and staging necessary
- Consult oncology team urgently

## ⚠️ Medical Disclaimer

**IMPORTANT: This AI system is designed for research and educational purposes only.**

### Key Limitations

❌ **NOT** for clinical diagnosis without physician oversight
❌ **NOT** FDA-approved for medical use
❌ **NOT** a replacement for trained gastroenterologists
❌ **NOT** validated for all colonoscopy imaging systems

### Required Actions

✅ **Always** consult qualified medical professionals
✅ **Always** verify AI findings with histopathology
✅ **Always** follow established clinical guidelines
✅ **Always** consider patient history and symptoms

### Training Limitations

- Trained on limited dataset (Kvasir-SEG)
- Performance may vary with different imaging equipment
- Requires validation in diverse clinical settings
- May not detect rare or unusual polyp presentations

**For medical diagnosis and treatment decisions, consult board-certified gastroenterologists and pathologists.**

## 🎓 Training Your Own Model

### YOLO Training Notebook
```bash
jupyter notebook Clinical_Polyp_Classification_Segmentation.ipynb
```

**Training Steps:**
1. Prepare dataset in YOLO format (images + labels)
2. Configure `data.yaml` with class names and paths
3. Run training with `yolo train` command
4. Export best weights to `weights/best.pt`
5. Update `MODEL_PATH` in `yolo_api_server.py`

### Training Configuration
```yaml
# data.yaml
path: /path/to/yolo_clinical_dataset
train: images/train
val: images/val
nc: 4
names: ['Normal', 'Hyperplastic', 'Adenomatous', 'Adenocarcinoma']
```

## 🧪 Testing

### Backend API Testing
```bash
# Test health endpoint
curl http://localhost:8000/api/health

# Test detection endpoint
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/api/detect
```

### Frontend Testing
```bash
cd clinical-view-frontend
npm run build  # Test production build
npm run preview  # Preview production build
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Areas for Improvement
- [ ] Add more diverse colonoscopy datasets
- [ ] Implement real-time video analysis
- [ ] Add multi-language support
- [ ] Improve model accuracy with ensemble methods
- [ ] Add user authentication and session management
- [ ] Implement DICOM image support
- [ ] Add integration with PACS systems
- [ ] Mobile application development

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Known Issues

- Large image files (>10MB) may cause slow processing
- Backend requires GPU for optimal performance
- Mobile responsiveness needs improvement
- PDF export may not work in all browsers

## 📦 Dependencies

### Backend (Python)
```txt
fastapi==0.104.1
uvicorn==0.24.0
ultralytics==8.0.196
opencv-python==4.8.1.78
pillow==10.1.0
python-multipart==0.0.6
```

### Frontend (Node.js)
```json
{
  "react": "^18.3.1",
  "typescript": "^5.6.2",
  "vite": "^5.4.2",
  "tailwindcss": "^3.4.1",
  "@radix-ui": "latest",
  "lucide-react": "latest"
}
```

## 🚀 Deployment

### Docker Deployment (Coming Soon)
```bash
# Build Docker image
docker build -t polyp-detection .

# Run container
docker run -p 8000:8000 -p 5173:5173 polyp-detection
```

### Cloud Deployment Options
- **AWS**: EC2 + S3 + CloudFront
- **Azure**: App Service + Blob Storage
- **Google Cloud**: Cloud Run + Cloud Storage
- **Heroku**: Web + Worker dynos

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses
- **YOLO**: AGPL-3.0 (Ultralytics)
- **React**: MIT License
- **FastAPI**: MIT License
- **Kvasir-SEG Dataset**: CC BY 4.0

## 🙏 Acknowledgments

- **Kvasir-SEG Dataset**: Jha et al., 2020 - Comprehensive polyp segmentation dataset
- **Ultralytics YOLO**: State-of-the-art object detection framework
- **PyTorch**: Deep learning framework by Meta AI
- **FastAPI**: Modern, high-performance web framework
- **React**: UI library by Meta
- **shadcn/ui**: Beautiful component library
- **Vite**: Next-generation frontend tooling

## 📚 References

### Academic Papers
1. Jha, D., et al. (2020). **Kvasir-SEG: A Segmented Polyp Dataset**. MMM 2020.
2. Ultralytics (2023). **YOLOv8: A New State-of-the-Art Computer Vision Model**.
3. Ronneberger, O., et al. (2015). **U-Net: Convolutional Networks for Biomedical Image Segmentation**.

### Documentation
- [YOLO Documentation](https://docs.ultralytics.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vitejs.dev/)

### Datasets
- [Kvasir-SEG Dataset](https://datasets.simula.no/kvasir-seg/)
- [PolypGen Dataset](https://www.synapse.org/#!Synapse:syn26707219)

## 📧 Contact & Support

### Repository Information
- **GitHub**: [S-PANDIYAN/AI-Based-Polyp-Detection-System](https://github.com/S-PANDIYAN/AI-Based-Polyp-Detection-System)
- **Issues**: [Report bugs or request features](https://github.com/S-PANDIYAN/AI-Based-Polyp-Detection-System/issues)

### Get Help
- 💬 **GitHub Discussions**: Ask questions and share ideas
- 🐛 **Bug Reports**: Create an issue with detailed description
- 💡 **Feature Requests**: Suggest new features or improvements
- 📧 **Email**: Contact repository owner for collaboration

## 📊 Project Statistics

![GitHub stars](https://img.shields.io/github/stars/S-PANDIYAN/AI-Based-Polyp-Detection-System?style=social)
![GitHub forks](https://img.shields.io/github/forks/S-PANDIYAN/AI-Based-Polyp-Detection-System?style=social)
![GitHub issues](https://img.shields.io/github/issues/S-PANDIYAN/AI-Based-Polyp-Detection-System)
![GitHub license](https://img.shields.io/github/license/S-PANDIYAN/AI-Based-Polyp-Detection-System)

## 🎯 Roadmap

### Version 2.0 (Planned)
- [ ] Real-time video colonoscopy analysis
- [ ] Multi-model ensemble for improved accuracy
- [ ] DICOM image format support
- [ ] Integration with hospital PACS systems
- [ ] Mobile application (iOS/Android)
- [ ] Cloud-based deployment options
- [ ] Advanced analytics dashboard
- [ ] User authentication and role management

### Version 3.0 (Future)
- [ ] 3D polyp reconstruction
- [ ] AI-assisted biopsy recommendations
- [ ] Longitudinal patient tracking
- [ ] Federated learning for privacy-preserving training
- [ ] Multi-center validation studies

---

## 🌟 Star History

⭐ **Star this repository** if you find it helpful!

---

**Version**: 2.0.0  
**Last Updated**: January 19, 2026  
**Status**: Active Development  
**Maintained**: ✅ Yes

---

<div align="center">

### Built with ❤️ for Medical AI Research

**[⬆ Back to Top](#-ai-based-polyp-detection-system)**

</div>
