"""
Polyp Detection Web Application
A Streamlit-based frontend for AI-powered polyp detection in colonoscopy images
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import io
from pathlib import Path
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="AI Polyp Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
    }
    .polyp-detected {
        background-color: #ffcccc;
        border: 2px solid #ff0000;
        color: #cc0000;
    }
    .no-polyp {
        background-color: #ccffcc;
        border: 2px solid #00cc00;
        color: #006600;
    }
    .uncertain {
        background-color: #fff4cc;
        border: 2px solid #ffaa00;
        color: #cc8800;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
        color: #1a1a1a;
        font-size: 0.95rem;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
        color: #1a1a1a;
        font-size: 0.95rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# CNN Model Architecture (must match the trained model)
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        
        # Block 1 - Two Conv layers
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.4)  # Updated to match trained model
        
        # Block 2 - Two Conv layers
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.4)  # Updated to match trained model
        
        # Block 3 - Two Conv layers
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.4)  # Updated to match trained model
        
        # Block 4 - Two Conv layers
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(0.4)  # Updated to match trained model
        
        # Dense Layers
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop_fc1 = nn.Dropout(0.6)  # Updated to match trained model
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.drop_fc2 = nn.Dropout(0.6)  # Updated to match trained model
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, x):
        # Block 1
        x = torch.relu(self.bn1_1(self.conv1_1(x)))
        x = torch.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Block 2
        x = torch.relu(self.bn2_1(self.conv2_1(x)))
        x = torch.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Block 3
        x = torch.relu(self.bn3_1(self.conv3_1(x)))
        x = torch.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.drop3(x)
        
        # Block 4
        x = torch.relu(self.bn4_1(self.conv4_1(x)))
        x = torch.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        x = self.drop4(x)
        
        # Flatten and Dense
        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc1(x)
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.drop_fc2(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x


@st.cache_resource
def load_model(model_path):
    """Load the trained PyTorch model with caching"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = CNNClassifier()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def preprocess_image(image, img_size=256):
    """
    Preprocess image for model input
    
    IMPORTANT: PIL loads images in RGB format, which is what the model was trained on.
    No color space conversion needed.
    """
    # Convert PIL Image to numpy array (already in RGB from PIL)
    img_array = np.array(image)
    
    # Resize image
    img_resized = cv2.resize(img_array, (img_size, img_size))
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert to PyTorch tensor format (C, H, W)
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor


def predict(model, image_tensor, device, threshold=0.50):
    """
    Make prediction on preprocessed image
    
    Binary Classification:
    - Class 0: Normal (healthy tissue)
    - Class 1: Polyp (abnormal tissue)
    
    Model trained with anti-overfitting techniques for good generalization.
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probability = output.item()
        
        # Adjusted threshold: Higher threshold = fewer false positives
        prediction = 1 if probability >= threshold else 0
        
        # Determine confidence level based on distance from decision boundary
        if probability >= 0.90 or probability <= 0.10:
            confidence_level = "Very High"
        elif probability >= 0.75 or probability <= 0.25:
            confidence_level = "High"
        elif probability >= 0.60 or probability <= 0.40:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"
    
    return prediction, probability, confidence_level


def create_visualization(image, prediction, probability, threshold):
    """Create visualization with prediction overlay"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original Image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Prediction Overlay
    axes[1].imshow(image)
    
    if prediction == 1:
        title = f'POLYP DETECTED\nScore: {probability*100:.1f}% (Threshold: {threshold*100:.0f}%)'
        color = 'red'
        axes[1].add_patch(plt.Rectangle((10, 10), image.width-20, image.height-20,
                                       fill=False, edgecolor=color, linewidth=4))
    elif probability >= 0.50:
        title = f'UNCERTAIN\nScore: {probability*100:.1f}% (Below {threshold*100:.0f}% threshold)'
        color = 'orange'
    else:
        title = f'NO POLYP\nScore: {probability*100:.1f}%'
        color = 'green'
    
    axes[1].set_title(title, fontsize=14, fontweight='bold', color=color)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close()
    
    return buf


def create_confidence_chart(probability, threshold):
    """Create a confidence meter visualization"""
    fig, ax = plt.subplots(figsize=(10, 2.5))
    
    # Create horizontal bar with interpretation zones
    polyp_prob = probability * 100
    no_polyp_prob = (1 - probability) * 100
    
    # Color based on confidence zones
    if probability >= 0.85:
        polyp_color = '#d32f2f'  # Dark red - High confidence
    elif probability >= 0.70:
        polyp_color = '#f57c00'  # Orange - Moderate confidence
    elif probability >= 0.50:
        polyp_color = '#fbc02d'  # Yellow - Low confidence
    else:
        polyp_color = '#c8e6c9'  # Light green - Very low
    
    ax.barh([''], [polyp_prob], color=polyp_color, label=f'Polyp Score: {polyp_prob:.1f}%', height=0.6)
    ax.barh([''], [no_polyp_prob], left=[polyp_prob], color='#66bb6a', 
            label=f'Normal Score: {no_polyp_prob:.1f}%', height=0.6)
    
    # Add threshold line
    threshold_x = threshold * 100
    ax.axvline(x=threshold_x, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
    ax.text(threshold_x, 0.35, f'Threshold: {threshold_x:.0f}%', 
            rotation=0, va='bottom', ha='center', fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))
    
    ax.set_xlim(0, 100)
    ax.set_xlabel('Confidence Score (%)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=2, fontsize=11)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add percentage text on bars
    if polyp_prob > 10:
        text_color = 'white' if polyp_prob > 30 else 'black'
        ax.text(polyp_prob/2, 0, f'{polyp_prob:.1f}%', 
                ha='center', va='center', fontweight='bold', fontsize=13, color=text_color)
    if no_polyp_prob > 10:
        ax.text(polyp_prob + no_polyp_prob/2, 0, f'{no_polyp_prob:.1f}%', 
                ha='center', va='center', fontweight='bold', fontsize=13, color='white')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close()
    
    return buf


def main():
    # Header
    st.markdown('<div class="main-header">🏥 AI Polyp Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Deep Learning for Colonoscopy Image Analysis</div>', 
                unsafe_allow_html=True)
    
    # Important Notice
    st.markdown("""
    <div class="info-box">
    <strong>✅ Binary Classification Model:</strong><br>
    This model was trained on <strong>BOTH classes</strong>:<br>
    • <strong>Class 0 (Normal):</strong> ~700 healthy cecum images<br>
    • <strong>Class 1 (Polyp):</strong> ~880 polyp images from Kvasir-SEG<br><br>
    The model uses <strong>advanced anti-overfitting techniques</strong> including strong data augmentation,
    dropout regularization, and L2 weight decay to ensure it <strong>generalizes well to new images</strong>.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/stomach.png", width=100)
        st.title("⚙️ Settings")
        
        st.markdown("### Detection Threshold")
        
        # Threshold slider
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.50,
            max_value=0.95,
            value=0.50,
            step=0.05,
            help="Probability threshold for polyp classification"
        )
        
        st.markdown(f"""
        **Current: {threshold*100:.0f}%**
        
        - **50%**: Standard (balanced) ✅
        - **60-70%**: Conservative
        - **80-95%**: High specificity
        
        <small>Model trained on balanced dataset of Normal + Polyp images</small>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.title("📋 Model Information")
        
        st.markdown("""
        ### Architecture
        - **Type:** Custom CNN (4 blocks)
        - **Parameters:** ~14M
        - **Framework:** PyTorch 2.x
        
        ### Training Data
        - **Class 0 (Normal):** ~700 images
        - **Class 1 (Polyp):** ~880 images
        - **Total:** ~1,580 images (balanced)
        
        ### Anti-Overfitting
        - ✅ Dropout (0.4 conv, 0.6 dense)
        - ✅ L2 Weight Decay (1e-4)
        - ✅ Strong Data Augmentation
        - ✅ Learning Rate Scheduling
        - ✅ Early Stopping
        """)
        
        st.markdown("---")
        
        # Model status
        model_path = Path("models/cnn_best_model.pth")
        if model_path.exists():
            st.success("✅ Model Loaded")
            import os
            import time
            file_size = model_path.stat().st_size / (1024*1024)
            modified_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(model_path)))
            st.info(f"📦 **Size:** {file_size:.1f} MB")
            st.info(f"📅 **Last trained:** {modified_time}")
            st.success("🎯 **Most Recent Model**")
        else:
            st.error("❌ Model Not Found")
        
        # Device info
        device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        st.info(f"🖥️ Device: {device_name}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a colonoscopy image...",
            type=['png', 'jpg', 'jpeg', 'bmp']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', width='stretch')
            
            st.markdown(f"""
            <div class="info-box">
            <strong>📊 Image Info:</strong><br>
            📐 Size: {image.width} x {image.height}px<br>
            💾 File: {uploaded_file.size / 1024:.2f} KB
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🔍 Analysis Results")
        
        if uploaded_file is not None:
            if st.button("🚀 Analyze Image", type="primary", width='stretch'):
                with st.spinner("🔄 Analyzing..."):
                    model, device = load_model("models/cnn_best_model.pth")
                    
                    if model is not None:
                        img_tensor = preprocess_image(image)
                        prediction, probability, confidence_level = predict(model, img_tensor, device, threshold)
                        
                        # Display result
                        if prediction == 1:
                            st.markdown(f"""
                            <div class="prediction-box polyp-detected">
                            ⚠️ POLYP DETECTED<br>
                            <small>{confidence_level} Confidence ({probability*100:.1f}%)</small>
                            </div>
                            """, unsafe_allow_html=True)
                            st.error("**Medical attention required**")
                            st.warning("Consult with a gastroenterologist for diagnosis.")
                            
                        elif probability >= 0.50:
                            st.markdown(f"""
                            <div class="prediction-box uncertain">
                            ⚠️ UNCERTAIN<br>
                            <small>Score: {probability*100:.1f}% (Below threshold)</small>
                            </div>
                            """, unsafe_allow_html=True)
                            st.warning("**Inconclusive - Specialist review recommended**")
                            
                        else:
                            st.markdown(f"""
                            <div class="prediction-box no-polyp">
                            ✅ NO POLYP DETECTED<br>
                            <small>{confidence_level} Confidence ({probability*100:.1f}%)</small>
                            </div>
                            """, unsafe_allow_html=True)
                            st.success("Negative result")
                        
                        # Metrics
                        st.markdown("---")
                        st.subheader("📊 Confidence Analysis")
                        
                        confidence_chart = create_confidence_chart(probability, threshold)
                        st.image(confidence_chart, width='stretch')
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric("Polyp Score", f"{probability*100:.2f}%")
                        with metric_col2:
                            st.metric("Threshold", f"{threshold*100:.0f}%")
                        with metric_col3:
                            st.metric("Confidence", confidence_level)
                        
                        # Visualization
                        st.markdown("---")
                        st.subheader("🖼️ Visual Analysis")
                        
                        viz_buffer = create_visualization(image, prediction, probability, threshold)
                        st.image(viz_buffer, width='stretch')
                        
                        # Download
                        st.markdown("---")
                        st.subheader("💾 Export Results")
                        
                        col_dl1, col_dl2 = st.columns(2)
                        
                        with col_dl1:
                            viz_buffer.seek(0)
                            st.download_button(
                                label="📥 Download Analysis",
                                data=viz_buffer,
                                file_name=f"polyp_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                width='stretch'
                            )
                        
                        with col_dl2:
                            report = f"""POLYP DETECTION REPORT
{'='*50}

Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

RESULT: {('POLYP DETECTED' if prediction == 1 else 'UNCERTAIN' if probability >= 0.50 else 'NO POLYP')}

SCORES:
- Polyp Probability: {probability*100:.2f}%
- Normal Probability: {(1-probability)*100:.2f}%
- Confidence Level: {confidence_level}
- Decision Threshold: {threshold*100:.0f}%

IMAGE:
- Size: {image.width}x{image.height}px
- File: {uploaded_file.name}

MODEL DETAILS:
- Architecture: CNN (~14M parameters)
- Training: Binary Classification
  * Class 0 (Normal): ~700 images
  * Class 1 (Polyp): ~880 images
- Regularization: Dropout + L2 + Data Augmentation
- Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}

ANTI-OVERFITTING MEASURES:
- Strong data augmentation (rotation, scaling, color jitter, etc.)
- Dropout regularization (0.4 conv, 0.6 dense)
- L2 weight decay
- Learning rate scheduling
- Early stopping

DISCLAIMER:
This is AI-assisted analysis for research/educational purposes.
Always consult qualified healthcare providers for medical diagnosis
and treatment decisions. This tool does not replace professional
medical judgment.
"""
                            st.download_button(
                                label="📄 Download Report",
                                data=report,
                                file_name=f"polyp_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                width='stretch'
                            )
                    else:
                        st.error("Failed to load model")
        else:
            st.info("👆 Upload an image to begin")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>⚕️ Medical Disclaimer</strong></p>
    <p>This AI system is for <strong>research and educational purposes only</strong>. 
    The model was trained on a balanced dataset of Normal and Polyp images using
    advanced anti-overfitting techniques. However, it should <strong>NOT be used as 
    the sole diagnostic tool</strong>. Always consult qualified healthcare providers 
    for medical diagnosis and treatment decisions.</p>
    <p style='margin-top: 15px;'><strong>Model:</strong> Binary CNN Classifier | 
    <strong>Training:</strong> ~1,580 images (Normal + Polyp) | 
    <strong>File:</strong> models/cnn_best_model.pth</p>
    <p style='margin-top: 20px;'><em>Powered by PyTorch | Built with Streamlit | © 2026</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
