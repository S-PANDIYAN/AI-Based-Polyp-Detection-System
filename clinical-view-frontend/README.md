# Clinical View Pro - Polyp Detection System

AI-powered polyp detection and classification system using YOLO deep learning models for real-time colonoscopy analysis.

## Features

- 🔍 **Multi-Class Detection**: Detects and classifies polyps (Normal, Hyperplastic, Adenomatous, Adenocarcinoma)
- 🎯 **High Accuracy**: YOLO-based model with real-time inference
- 📊 **Risk Assessment**: Automatic risk level calculation (Low, Medium, High, Critical)
- 🖼️ **Visual Analysis**: Side-by-side comparison with annotated detections
- 📈 **Confidence Scoring**: Detailed confidence metrics for each detection
- 📋 **Export Reports**: Generate PDF reports for clinical documentation
- 🎨 **Modern UI**: Clean, professional interface built with React + TypeScript

## Project Structure

```
clinical-view-frontend/
├── src/
│   ├── components/        # React components
│   ├── pages/             # Page components
│   ├── services/          # API services
│   └── lib/               # Utilities
├── yolo_api_server.py     # FastAPI backend server
├── requirements.txt       # Python dependencies
├── package.json           # Node dependencies
└── start_all.bat          # Windows startup script
```

## Prerequisites

- **Python 3.8+** with pip
- **Node.js 18+** with npm
- **YOLO Model**: Trained model weights (best.pt)

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Node.js Dependencies

```bash
npm install
```

### 3. Configure Environment

Create a `.env` file:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_MOCK_API=false
```

## Running the Application

### Option 1: Use the Startup Script (Windows)

Double-click `start_all.bat` or run:

```bash
start_all.bat
```

This will start both the backend API and frontend dev server.

### Option 2: Manual Start

**Terminal 1 - Start Backend API:**

```bash
python yolo_api_server.py
```

**Terminal 2 - Start Frontend:**

```bash
npm run dev
```

## Accessing the Application

- **Frontend UI**: http://localhost:5173
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## API Endpoints

### Health Check
```
GET /api/health
```

### Polyp Detection
```
POST /api/detect
Content-Type: multipart/form-data
Body: image file

Response:
{
  "detections": [...],
  "annotatedImage": "base64_string",
  "totalDetections": 2,
  "processingTime": 0.4,
  "confidenceStats": {...}
}
```

## Model Configuration

The API server looks for the trained YOLO model at:
- `../yolo_clinical_training/polyp_multiclass/weights/best.pt`

If not found, it falls back to:
- `../yolo11n.pt` (base YOLO model)

Update `MODEL_PATH` in `yolo_api_server.py` to point to your model.

## Usage

1. **Upload Image**: Click or drag-and-drop a colonoscopy image
2. **Analyze**: Click "Run Detection Analysis" 
3. **Review Results**: View annotated image with detected polyps
4. **Export**: Generate PDF report with findings

## Technologies

**Frontend:**
- React 18
- TypeScript
- Vite
- Tailwind CSS
- Shadcn/ui Components

**Backend:**
- FastAPI
- Ultralytics YOLO
- OpenCV
- PyTorch

## License

This project is for educational and research purposes.

- Edit files directly within the Codespace and commit and push your changes once you're done.

## What technologies are used for this project?

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## How can I deploy this project?

Simply open [Lovable](https://lovable.dev/projects/REPLACE_WITH_PROJECT_ID) and click on Share -> Publish.

## Can I connect a custom domain to my Lovable project?

Yes, you can!

To connect a domain, navigate to Project > Settings > Domains and click Connect Domain.

Read more here: [Setting up a custom domain](https://docs.lovable.dev/features/custom-domain#custom-domain)
