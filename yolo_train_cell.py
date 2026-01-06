# This content will be added to the notebook
# YOLOv8 Training Cell

"""
# Train YOLOv8 on Polyp Dataset
results = yolo_model.train(
    data=yaml_path,
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    save=True,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    
    # Optimization parameters for efficiency
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    
    # Data augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    
    # Model saving
    name='polyp_yolov8n',
    exist_ok=True,
    pretrained=True,
    
    # Performance
    workers=8,
    cache='ram',
    amp=True,  # Automatic Mixed Precision for faster training
    
    # Validation
    val=True,
    plots=True,
    save_period=10
)

print("="*70)
print("✅ Training Complete!")
print(f"📊 Best mAP@50: {results.results_dict['metrics/mAP50(B)']:.4f}")
print(f"📂 Model saved to: runs/detect/polyp_yolov8n/weights/best.pt")
print("="*70)

# Load best model for inference
best_model = YOLO('runs/detect/polyp_yolov8n/weights/best.pt')
"""
