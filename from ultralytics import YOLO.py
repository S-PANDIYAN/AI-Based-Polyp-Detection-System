from ultralytics import YOLO

# Load best model
model = YOLO(r'D:\PolyP\yolo_clinical_training\polyp_multiclass\weights\best.pt')

# Run inference
results = model.predict('your_image.png', conf=0.25)