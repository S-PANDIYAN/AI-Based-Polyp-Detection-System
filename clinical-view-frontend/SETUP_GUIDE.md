# Setup Guide - Clinical View Pro

Complete setup instructions for the Polyp Detection System.

## Quick Start (Windows)

1. **Install Dependencies**
   ```bash
   cd d:\PolyP\clinical-view-frontend
   npm install
   pip install -r requirements.txt
   ```

2. **Start the Application**
   ```bash
   start_all.bat
   ```

3. **Access the Application**
   - Frontend: http://localhost:5173
   - API Docs: http://localhost:8000/docs

## Detailed Setup Instructions

### Prerequisites

1. **Python 3.8 or higher**
   - Download from: https://www.python.org/downloads/
   - Make sure to check "Add Python to PATH" during installation

2. **Node.js 18 or higher**
   - Download from: https://nodejs.org/
   - LTS version recommended

3. **Git** (optional, for version control)
   - Download from: https://git-scm.com/

### Step-by-Step Installation

#### 1. Frontend Setup

```bash
# Navigate to the frontend directory
cd d:\PolyP\clinical-view-frontend

# Install Node.js dependencies
npm install

# This will install:
# - React, TypeScript, Vite
# - Tailwind CSS, Radix UI components
# - React Query for data fetching
# - All other frontend dependencies
```

#### 2. Backend Setup

```bash
# In the same directory (clinical-view-frontend)
# Install Python dependencies
pip install -r requirements.txt

# This will install:
# - FastAPI (web framework)
# - Ultralytics YOLO (detection model)
# - OpenCV (image processing)
# - PyTorch (deep learning)
# - Other required packages
```

#### 3. Model Configuration

The system needs a trained YOLO model. The API server looks for:

**Primary location:**
```
d:\PolyP\yolo_clinical_training\polyp_multiclass\weights\best.pt
```

**Fallback location:**
```
d:\PolyP\yolo11n.pt
```

If you have a custom model path, edit `yolo_api_server.py`:

```python
MODEL_PATH = Path("your/custom/path/to/model.pt")
```

### Running the Application

#### Method 1: Automated Startup (Recommended)

Double-click `start_all.bat` or run in PowerShell:

```powershell
.\start_all.bat
```

This will:
1. Start the FastAPI backend on port 8000
2. Start the Vite dev server on port 5173
3. Open two terminal windows (one for each service)

#### Method 2: Manual Startup

**Terminal 1 - Backend API:**
```bash
cd d:\PolyP\clinical-view-frontend
python yolo_api_server.py
```

**Terminal 2 - Frontend:**
```bash
cd d:\PolyP\clinical-view-frontend
npm run dev
```

### Verification

#### 1. Check Backend Health

Open browser to: http://localhost:8000/api/health

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "path/to/your/model.pt"
}
```

#### 2. Check Frontend

Open browser to: http://localhost:5173

You should see the Clinical View Pro dashboard.

#### 3. Test Detection

1. Click "New Scan" or upload area
2. Upload a colonoscopy image
3. Click "Run Detection Analysis"
4. View results with annotated detections

### Environment Configuration

Create or edit `.env` file in `clinical-view-frontend`:

```env
# API Configuration
VITE_API_BASE_URL=http://localhost:8000

# Mock Mode (set to 'true' for testing without backend)
VITE_MOCK_API=false
```

**Note:** After changing `.env`, restart the frontend server.

### Troubleshooting

#### Issue: "Model not found"

**Solution:**
1. Check if model file exists at the path specified in `yolo_api_server.py`
2. Update `MODEL_PATH` variable to correct location
3. The system will use a base YOLO model if trained model is not found

#### Issue: "Port already in use"

**Backend (port 8000):**
```python
# In yolo_api_server.py, change:
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Changed from 8000
```

**Frontend (port 5173):**
```typescript
// In vite.config.ts, add:
server: {
  port: 5174  // Changed from 5173
}
```

#### Issue: CORS errors

Make sure backend `CORSMiddleware` allows your frontend origin:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Or "*" for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### Issue: Module not found

**Python:**
```bash
pip install --upgrade -r requirements.txt
```

**Node.js:**
```bash
npm install
```

#### Issue: Image upload fails

1. Check backend logs for errors
2. Verify image format (JPG, PNG supported)
3. Check file size (< 10MB recommended)
4. Review browser console for errors

### Building for Production

#### Frontend

```bash
npm run build
```

Output will be in `dist/` directory.

#### Backend

The Python backend can run as-is in production. For production deployment:

```bash
uvicorn yolo_api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Performance Optimization

1. **GPU Acceleration**: Install CUDA-enabled PyTorch for faster inference
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Model Optimization**: Use smaller YOLO models for faster inference
   - YOLOv8n (fastest, least accurate)
   - YOLOv8s (balanced)
   - YOLOv8m (more accurate, slower)

3. **Image Preprocessing**: Resize large images before upload

### Project Structure

```
clinical-view-frontend/
├── src/
│   ├── components/
│   │   ├── dashboard/       # Dashboard components
│   │   └── ui/              # Reusable UI components
│   ├── pages/
│   │   └── Index.tsx        # Main page with detection logic
│   ├── services/
│   │   └── polypDetectionAPI.ts  # API client
│   ├── hooks/               # Custom React hooks
│   └── lib/                 # Utility functions
├── public/                  # Static assets
├── yolo_api_server.py       # Backend API server
├── requirements.txt         # Python dependencies
├── package.json             # Node dependencies
├── .env                     # Environment configuration
├── start_all.bat           # Windows startup script
└── README.md               # Project documentation
```

### API Documentation

Once backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Security Notes

⚠️ **Important for Production:**

1. Change `allow_origins=["*"]` to specific frontend domain
2. Add authentication/authorization
3. Implement rate limiting
4. Use HTTPS
5. Sanitize uploaded files
6. Add input validation
7. Use environment variables for sensitive data

### Support

For issues:
1. Check backend terminal for Python errors
2. Check frontend terminal for build errors
3. Check browser console for JavaScript errors
4. Review network tab for API failures

### Next Steps

1. Upload test images to verify detection
2. Adjust sensitivity threshold in sidebar
3. Export reports as PDF
4. Customize UI theme and styling
5. Add additional features as needed

## Common Commands

```bash
# Frontend
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint

# Backend
python yolo_api_server.py           # Start API server
uvicorn yolo_api_server:app --reload  # Start with auto-reload

# Both
start_all.bat        # Start both servers (Windows)
```

## Configuration Files

- `.env` - Environment variables
- `vite.config.ts` - Vite configuration
- `tailwind.config.ts` - Tailwind CSS configuration
- `tsconfig.json` - TypeScript configuration
- `requirements.txt` - Python dependencies
- `package.json` - Node dependencies

---

**Ready to go!** 🚀

If you encounter any issues, refer to the troubleshooting section or check the logs.
