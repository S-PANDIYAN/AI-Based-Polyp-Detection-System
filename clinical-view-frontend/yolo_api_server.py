"""
YOLO Multi-Class Polyp Detection API Server
Provides REST API for the React frontend
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
from typing import List, Dict
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YOLO Polyp Detection API",
    description="API for multi-class polyp detection using YOLO",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model path - point to your trained YOLO model
MODEL_PATH = Path("../yolo_clinical_training/polyp_multiclass/weights/best.pt")

# Class mappings for polyp types (matching training notebook)
# 'Normal': 0, 'Hyperplasia': 1, 'Adenoma': 2, 'Adenocarcinoma': 3
CLASS_NAMES = {
    0: "Normal",
    1: "Hyperplastic Polyp",
    2: "Adenomatous Polyp",
    3: "Adenocarcinoma"
}

# Risk level mapping based on polyp type
RISK_LEVELS = {
    "Normal": "low",
    "Hyperplastic Polyp": "low",
    "Adenomatous Polyp": "high",
    "Adenocarcinoma": "critical"
}

# Load model
model = None

def load_yolo_model():
    """Load YOLO model"""
    global model
    try:
        if not MODEL_PATH.exists():
            logger.error(f"Model not found at {MODEL_PATH}")
            # Fallback to base model
            model_path = Path("../yolo11n.pt")
            if model_path.exists():
                logger.info(f"Using base model: {model_path}")
                model = YOLO(str(model_path))
            else:
                logger.warning("No YOLO model found, using default")
                model = YOLO("yolo11n.pt")  # Download if needed
        else:
            logger.info(f"Loading model from {MODEL_PATH}")
            model = YOLO(str(MODEL_PATH))
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_yolo_model()
    if not success:
        logger.warning("Failed to load model on startup")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH) if MODEL_PATH.exists() else "base model"
    }

def process_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded image bytes to numpy array"""
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    return np.array(image)

def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode image to base64 string"""
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Save to bytes
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=95)
    
    # Encode to base64
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def calculate_risk_level(class_name: str, confidence: float) -> str:
    """Calculate risk level based on class and confidence"""
    base_risk = RISK_LEVELS.get(class_name, "low")
    
    # Adjust based on confidence
    if confidence >= 0.8:
        return base_risk
    elif confidence >= 0.5:
        if base_risk == "high":
            return "medium"
        return base_risk
    else:
        return "low"

@app.post("/api/detect")
async def detect_polyps(image: UploadFile = File(...)):
    """
    Detect and classify polyps in the uploaded image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Track processing time
        start_time = time.time()
        
        # Read image
        image_bytes = await image.read()
        img_array = process_image(image_bytes)
        
        # Run inference with stricter NMS to prevent duplicate boxes
        results = model(
            img_array, 
            conf=0.25,      # Confidence threshold
            iou=0.3,        # Lower IoU = more aggressive NMS (removes more overlaps)
            max_det=10,     # Maximum 10 detections per image
            agnostic_nms=False  # Class-specific NMS
        )
        
        # Process results
        detections = []
        annotated_image = img_array.copy()
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for idx, box in enumerate(boxes):
                    # Extract box information
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Get class name
                    class_name = CLASS_NAMES.get(cls, f"Class {cls}")
                    
                    # Calculate bbox properties
                    x1, y1, x2, y2 = map(int, xyxy)
                    width = x2 - x1
                    height = y2 - y1
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    area = width * height
                    
                    # Calculate risk level
                    risk_level = calculate_risk_level(class_name, conf)
                    
                    # Create detection object
                    detection = {
                        "id": idx + 1,
                        "class": class_name,
                        "confidence": round(conf, 3),
                        "bbox": {
                            "x": x1,
                            "y": y1,
                            "width": width,
                            "height": height
                        },
                        "center": {
                            "x": center_x,
                            "y": center_y
                        },
                        "area": area,
                        "riskLevel": risk_level
                    }
                    detections.append(detection)
                    
                    # Draw on annotated image
                    color_map = {
                        "critical": (139, 0, 0),   # Dark Red
                        "high": (255, 0, 0),       # Red
                        "medium": (255, 165, 0),   # Orange
                        "low": (0, 255, 0)         # Green
                    }
                    color = color_map.get(risk_level, (0, 255, 0))
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label with background
                    label = f"{class_name} {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(
                        annotated_image,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0] + 10, y1),
                        color,
                        -1
                    )
                    cv2.putText(
                        annotated_image,
                        label,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Calculate confidence statistics
        confidence_stats = {"high": 0, "medium": 0, "low": 0}
        for det in detections:
            if det["confidence"] >= 0.8:
                confidence_stats["high"] += 1
            elif det["confidence"] >= 0.5:
                confidence_stats["medium"] += 1
            else:
                confidence_stats["low"] += 1
        
        # Encode annotated image
        annotated_base64 = encode_image_to_base64(annotated_image)
        
        # Prepare response
        response = {
            "detections": detections,
            "annotatedImage": annotated_base64,
            "processingTime": round(processing_time, 2),
            "imageSize": {
                "width": img_array.shape[1],
                "height": img_array.shape[0]
            },
            "totalDetections": len(detections),
            "confidenceStats": confidence_stats
        }
        
        logger.info(f"Processed image: {len(detections)} detections in {processing_time:.2f}s")
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
