# Polyp Detection System - Setup & Run Guide

## 🚀 Quick Start

### 1️⃣ Start Backend API (Flask + PyTorch)

```bash
# Install dependencies (one-time setup)
pip install -r requirements-api.txt

# Start Flask API server
python api_server.py
```

The API will run on: **http://localhost:5000**

---

### 2️⃣ Start Frontend (React + TypeScript)

```bash
# Navigate to frontend
cd frontend

# Install dependencies (one-time setup)
npm install

# Start development server
npm run dev
```

The frontend will run on: **http://localhost:5173**

---

## 🔌 API Endpoints

### Health Check
```
GET http://localhost:5000/api/health
```

### Predict Polyp
```
POST http://localhost:5000/api/predict
Body: FormData with 'image' file and 'threshold' (optional)
```

### Model Info
```
GET http://localhost:5000/api/model-info
```

---

## 📁 File Structure

```
D:\PolyP\
├── api_server.py              # Flask backend API
├── models/
│   └── cnn_best_model.pth     # PyTorch trained model
├── frontend/
│   ├── src/
│   │   ├── services/
│   │   │   └── api.ts         # API service layer
│   │   ├── hooks/
│   │   │   └── useAnalysis.ts # Analysis hook (now using real API)
│   │   └── pages/
│   │       └── Index.tsx      # Main page
│   └── .env                   # API URL configuration
└── requirements-api.txt       # Python dependencies
```

---

## ⚙️ Configuration

### Backend (api_server.py)
- **Port**: 5000
- **Model Path**: `models/cnn_best_model.pth`
- **CORS**: Enabled for frontend

### Frontend (.env)
- **API URL**: `http://localhost:5000/api`

---

## 🧪 Testing the Connection

1. Start backend: `python api_server.py`
2. Test health: Open `http://localhost:5000/api/health` in browser
3. Start frontend: `cd frontend && npm run dev`
4. Upload an image and click "Analyze"

---

## 🔧 Troubleshooting

### Backend Issues
- **Model not loading**: Check if `models/cnn_best_model.pth` exists
- **Port already in use**: Change port in `api_server.py`
- **CUDA errors**: Model will automatically fall back to CPU

### Frontend Issues
- **API connection failed**: Check if backend is running on port 5000
- **CORS errors**: Ensure Flask-CORS is installed
- **Build errors**: Run `npm install` in frontend folder

---

## 📦 Production Build

### Backend
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```

### Frontend
```bash
cd frontend
npm run build
# Deploy the 'dist' folder
```

---

## 🎯 Features

✅ Real-time polyp detection using PyTorch CNN  
✅ RESTful API with Flask  
✅ Modern React/TypeScript frontend  
✅ Automatic GPU acceleration (if available)  
✅ Adjustable detection threshold  
✅ Confidence scoring  
✅ Image analysis and visualization  

---

**© 2026 AI Polyp Detection System**
