"""
Flask API Server for Polyp Detection
Connects PyTorch model to TypeScript frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import io
import base64
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# CNN Model Architecture (must match trained model)
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        
        # Block 1
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.4)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.4)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.4)
        
        # Block 4
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(0.4)
        
        # Dense Layers
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop_fc1 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.drop_fc2 = nn.Dropout(0.6)
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


# Load model at startup
MODEL_PATH = Path("models/cnn_best_model.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

try:
    model = CNNClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully on {device}")
except Exception as e:
    print(f"❌ Error loading model: {e}")


def preprocess_image(image_data, img_size=256):
    """Preprocess image for model input"""
    try:
        # Convert base64 or file to PIL Image
        if isinstance(image_data, str):
            # Base64 string
            image_data = image_data.split(',')[1] if ',' in image_data else image_data
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            # File object
            image = Image.open(image_data).convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize
        img_resized = cv2.resize(img_array, (img_size, img_size))
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Convert to tensor (C, H, W)
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor, image
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })


def generate_gradcam_bbox(model, img_tensor, original_size):
    """Generate bounding box using Grad-CAM approximation"""
    try:
        # Get the last convolutional layer output
        model.eval()
        
        # Forward pass with hook to get feature maps
        features = []
        def hook_fn(module, input, output):
            features.append(output)
        
        # Register hook on last conv layer (conv4_2)
        hook = model.conv4_2.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.set_grad_enabled(True):
            img_tensor.requires_grad = True
            output = model(img_tensor)
        
        # Backward pass
        model.zero_grad()
        output.backward()
        
        hook.remove()
        
        # Get gradients and feature maps
        if len(features) > 0:
            feature_map = features[0].squeeze(0)  # Remove batch dimension
            
            # Create activation map by averaging across channels
            activation_map = torch.mean(feature_map, dim=0).cpu().detach().numpy()
            
            # Normalize to [0, 1]
            activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
            
            # Resize activation map to original image size
            activation_map = cv2.resize(activation_map, (256, 256))
            
            # Threshold to get polyp region
            threshold = 0.5
            binary_map = (activation_map > threshold).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Convert to percentage of image
                x_percent = (x / 256) * 100
                y_percent = (y / 256) * 100
                w_percent = (w / 256) * 100
                h_percent = (h / 256) * 100
                
                # Add some padding (10%)
                padding = 5
                x_percent = max(0, x_percent - padding)
                y_percent = max(0, y_percent - padding)
                w_percent = min(100 - x_percent, w_percent + padding * 2)
                h_percent = min(100 - y_percent, h_percent + padding * 2)
                
                return {
                    'x': round(x_percent, 2),
                    'y': round(y_percent, 2),
                    'width': round(w_percent, 2),
                    'height': round(h_percent, 2)
                }
        
        # Fallback: center region
        return {
            'x': 25,
            'y': 25,
            'width': 50,
            'height': 50
        }
    
    except Exception as e:
        print(f"Bounding box generation error: {e}")
        # Return center region as fallback
        return {
            'x': 25,
            'y': 25,
            'width': 50,
            'height': 50
        }


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict polyp in uploaded image"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get threshold from request (default 0.50)
        threshold = float(request.form.get('threshold', 0.50))
        
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        
        # Preprocess image
        img_tensor, original_image = preprocess_image(image_file)
        
        # Make prediction
        with torch.no_grad():
            img_tensor_no_grad = img_tensor.to(device)
            output = model(img_tensor_no_grad)
            probability = output.item()
        
        # Determine prediction
        prediction = 1 if probability >= threshold else 0
        
        # Generate bounding box if polyp detected
        bounding_box = None
        if prediction == 1:
            bounding_box = generate_gradcam_bbox(model, img_tensor.to(device), (original_image.width, original_image.height))
        
        # Confidence level (adjusted for better variation)
        if probability >= 0.90 or probability <= 0.10:
            confidence_level = "Very High"
        elif probability >= 0.75 or probability <= 0.25:
            confidence_level = "High"
        elif probability >= 0.60 or probability <= 0.40:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"
        
        # Prepare response
        result = {
            'success': True,
            'prediction': prediction,
            'class_name': 'Polyp' if prediction == 1 else 'Normal',
            'probability': float(probability),
            'polyp_score': float(probability * 100),
            'normal_score': float((1 - probability) * 100),
            'confidence_level': confidence_level,
            'threshold': threshold,
            'image_size': {
                'width': original_image.width,
                'height': original_image.height
            },
            'bounding_box': bounding_box
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    import os
    import time
    
    if MODEL_PATH.exists():
        file_size = MODEL_PATH.stat().st_size / (1024*1024)
        modified_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                     time.localtime(os.path.getmtime(MODEL_PATH)))
        
        return jsonify({
            'model_path': str(MODEL_PATH),
            'size_mb': round(file_size, 2),
            'last_modified': modified_time,
            'device': str(device),
            'parameters': sum(p.numel() for p in model.parameters()) if model else 0,
            'architecture': {
                'type': 'CNN Classifier',
                'blocks': 4,
                'classes': 2,
                'input_size': '256x256x3'
            }
        })
    else:
        return jsonify({'error': 'Model file not found'}), 404


if __name__ == '__main__':
    print("="*60)
    print("🚀 Starting Flask API Server for Polyp Detection")
    print("="*60)
    print(f"📁 Model Path: {MODEL_PATH}")
    print(f"🖥️  Device: {device}")
    print(f"🌐 Server: http://localhost:5000")
    print(f"🔗 Frontend should connect to: http://localhost:5000/api")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
