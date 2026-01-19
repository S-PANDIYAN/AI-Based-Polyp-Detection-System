"""
YOLO Multi-Class Polyp Detection - Streamlit Frontend
Domain: Medical Image Analysis
Task: Real-time polyp detection and classification
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import os
import torch
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="YOLO Polyp Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .detection-box {
        border: 2px solid #4CAF50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Class definitions
CLASS_NAMES = ['Normal', 'Hyperplastic', 'Adenomatous', 'Sessile']
CLASS_COLORS = {
    0: (0, 255, 0),      # Green - Normal
    1: (255, 255, 0),    # Yellow - Hyperplastic
    2: (255, 165, 0),    # Orange - Adenomatous
    3: (255, 0, 0)       # Red - Sessile (High-risk)
}

CLASS_DESCRIPTIONS = {
    'Normal': {
        'risk': 'No Risk',
        'description': 'Healthy colon tissue without abnormalities',
        'action': 'Continue routine screening'
    },
    'Hyperplastic': {
        'risk': 'Low Risk',
        'description': 'Benign growths with minimal cancer potential',
        'action': 'Regular monitoring recommended'
    },
    'Adenomatous': {
        'risk': 'Moderate Risk',
        'description': 'Pre-cancerous polyps that require removal',
        'action': 'Removal and follow-up required'
    },
    'Sessile': {
        'risk': 'High Risk',
        'description': 'Flat/sessile polyps with higher malignancy potential',
        'action': '⚠️ Immediate medical attention required'
    }
}

# Model path
MODEL_PATH = r"D:\PolyP\yolo_clinical_training\polyp_multiclass\weights\best.pt"

@st.cache_resource
def load_model():
    """Load YOLO model with caching"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model not found at: {MODEL_PATH}")
            st.info("Please ensure the model is trained and saved at the correct path.")
            return None
        
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_polyp(model, image, conf_threshold=0.25, iou_threshold=0.45):
    """Run YOLO inference on image with NMS to remove duplicate detections"""
    try:
        # Convert PIL to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Run inference with NMS
        results = model.predict(
            source=image_np,
            conf=conf_threshold,
            iou=iou_threshold,  # IoU threshold for NMS
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=False,
            max_det=10  # Maximum detections per image
        )
        
        detections = []
        annotated_image = image_np.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract box data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = CLASS_NAMES[class_id]
                
                # Store detection
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class': class_name,
                    'class_id': class_id,
                    'confidence': confidence
                })
                
                # Draw bounding box
                color = CLASS_COLORS[class_id]
                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                
                # Draw label background
                label = f"{class_name}: {confidence*100:.1f}%"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated_image, (int(x1), int(y1)-label_height-10), 
                            (int(x1)+label_width, int(y1)), color, -1)
                
                # Draw label text
                cv2.putText(annotated_image, label, (int(x1), int(y1)-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return detections, annotated_image
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return [], image_np

def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 YOLO Multi-Class Polyp Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
        st.title("⚙️ Settings")
        
        # Model info
        st.subheader("📊 Model Information")
        if os.path.exists(MODEL_PATH):
            st.success("✅ Model Loaded")
            st.info(f"**Architecture:** YOLOv8n\n**Classes:** 4\n**mAP50:** 97.4%")
        else:
            st.error("❌ Model Not Found")
        
        # GPU info
        if torch.cuda.is_available():
            st.success(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.info("💻 Running on CPU")
        
        st.divider()
        
        # Confidence threshold
        conf_threshold = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        # IoU threshold for NMS
        iou_threshold = st.slider(
            "🔄 IoU Threshold (NMS)",
            min_value=0.1,
            max_value=0.9,
            value=0.45,
            step=0.05,
            help="Higher values = more boxes kept. Lower values = remove more overlapping boxes"
        )
        
        st.divider()
        
        # Class information
        st.subheader("🏷️ Polyp Classes")
        for i, class_name in enumerate(CLASS_NAMES):
            color_hex = '#{:02x}{:02x}{:02x}'.format(*CLASS_COLORS[i][::-1])
            st.markdown(f"**Class {i}:** {class_name}")
            st.markdown(f'<div style="background-color:{color_hex}; height:10px; border-radius:5px;"></div>', 
                       unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Colonoscopy Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
            help="Upload a colonoscopy image for polyp detection"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
            
            # Image info
            st.info(f"📐 Image Size: {image.size[0]} x {image.size[1]} pixels")
    
    with col2:
        st.subheader("🎯 Detection Results")
        
        if uploaded_file is not None:
            # Load model
            model = load_model()
            
            if model is not None:
                with st.spinner("🔍 Analyzing image..."):
                    # Run prediction
                    detections, annotated_image = predict_polyp(model, image, conf_threshold, iou_threshold)
                    
                    # Display annotated image
                    st.image(annotated_image, caption="Detected Polyps", use_container_width=True)
                    
                    # Detection summary
                    st.metric("Total Detections", len(detections))
    
    # Detailed Results Section
    if uploaded_file is not None and 'detections' in locals():
        st.divider()
        st.subheader("📋 Detailed Analysis Report")
        
        if len(detections) > 0:
            # Create results dataframe
            results_df = pd.DataFrame([{
                'Detection #': idx + 1,
                'Polyp Type': det['class'],
                'Confidence': f"{det['confidence']*100:.1f}%",
                'Risk Level': CLASS_DESCRIPTIONS[det['class']]['risk'],
                'Bounding Box': f"({det['bbox'][0]}, {det['bbox'][1]}) - ({det['bbox'][2]}, {det['bbox'][3]})"
            } for idx, det in enumerate(detections)])
            
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Detailed findings
            st.subheader("🔬 Clinical Findings")
            
            for idx, detection in enumerate(detections):
                class_name = detection['class']
                class_info = CLASS_DESCRIPTIONS[class_name]
                confidence = detection['confidence'] * 100
                
                # Risk-based styling
                if class_name == 'Sessile':
                    box_class = "detection-box" 
                    risk_emoji = "🔴"
                elif class_name == 'Adenomatous':
                    box_class = "warning-box"
                    risk_emoji = "🟠"
                elif class_name == 'Hyperplastic':
                    box_class = "success-box"
                    risk_emoji = "🟡"
                else:
                    box_class = "success-box"
                    risk_emoji = "🟢"
                
                with st.expander(f"{risk_emoji} Detection #{idx+1}: {class_name} ({confidence:.1f}% confidence)", expanded=True):
                    col_a, col_b = st.columns([2, 1])
                    
                    with col_a:
                        st.markdown(f"**Type:** {class_name}")
                        st.markdown(f"**Risk Level:** {class_info['risk']}")
                        st.markdown(f"**Description:** {class_info['description']}")
                        st.markdown(f"**Recommended Action:** {class_info['action']}")
                    
                    with col_b:
                        st.metric("Confidence", f"{confidence:.1f}%")
                        st.markdown(f"**Location:**")
                        st.code(f"Top-Left: ({detection['bbox'][0]}, {detection['bbox'][1]})\n"
                               f"Bottom-Right: ({detection['bbox'][2]}, {detection['bbox'][3]})")
            
            # Risk summary
            st.divider()
            st.subheader("⚠️ Risk Assessment Summary")
            
            high_risk = sum(1 for d in detections if d['class'] == 'Sessile')
            moderate_risk = sum(1 for d in detections if d['class'] == 'Adenomatous')
            low_risk = sum(1 for d in detections if d['class'] == 'Hyperplastic')
            normal = sum(1 for d in detections if d['class'] == 'Normal')
            
            risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
            
            with risk_col1:
                st.metric("🔴 High Risk", high_risk, help="Sessile polyps")
            with risk_col2:
                st.metric("🟠 Moderate Risk", moderate_risk, help="Adenomatous polyps")
            with risk_col3:
                st.metric("🟡 Low Risk", low_risk, help="Hyperplastic polyps")
            with risk_col4:
                st.metric("🟢 Normal", normal, help="Normal tissue")
            
            # Overall recommendation
            if high_risk > 0:
                st.error("⚠️ **URGENT:** High-risk polyps detected. Immediate medical consultation required.")
            elif moderate_risk > 0:
                st.warning("⚠️ **ATTENTION:** Moderate-risk polyps detected. Schedule removal procedure.")
            elif low_risk > 0:
                st.info("ℹ️ **NOTE:** Low-risk polyps detected. Continue regular monitoring.")
            else:
                st.success("✅ **HEALTHY:** No concerning polyps detected.")
        
        else:
            st.success("✅ No polyps detected in the image.")
            st.info("This could indicate healthy colon tissue or may require manual review by a specialist.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>⚕️ Medical Disclaimer:</strong> This AI system is designed to assist medical professionals 
        and should not replace expert clinical judgment. All findings should be verified by qualified healthcare providers.</p>
        <p>Model Performance: mAP50 = 97.4% | Precision = 97.5% | Recall = 92.8%</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
