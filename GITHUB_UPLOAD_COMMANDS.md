# Commands to Upload to GitHub

## Step 1: Add All Files (Respecting .gitignore)

```bash
cd d:\PolyP
git add .
git status
```

This will add all files EXCEPT:
- ❌ archive/
- ❌ dataset/
- ❌ cecum/
- ❌ test_samples/
- ❌ yolo_dataset/
- ❌ yolo_clinical_dataset/
- ❌ yolo_clinical_training/
- ❌ node_modules/
- ❌ Large model files (*.pth, *.pt except yolo11n.pt, yolov8n.pt)

## Step 2: Commit the Changes

```bash
git commit -m "Add clinical polyp detection system with React frontend and YOLO backend"
```

## Step 3: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `polyp-detection-system` (or your preferred name)
3. Description: "AI-powered polyp detection using YOLO and React"
4. Keep it **Public** or **Private** (your choice)
5. **DO NOT** initialize with README (you already have one)
6. Click "Create repository"

## Step 4: Link and Push to GitHub

```bash
# If you don't have a remote yet, add it:
git remote add origin https://github.com/YOUR_USERNAME/polyp-detection-system.git

# Or if remote already exists, update it:
git remote set-url origin https://github.com/YOUR_USERNAME/polyp-detection-system.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Alternative: If Already Have Remote

```bash
# Just push your changes
git push origin master
```

## Verify What Will Be Uploaded

Before committing, check what will be uploaded:

```bash
git status
```

## Check Repository Size (Important!)

```bash
# Check total size
git count-objects -vH
```

GitHub has a 100 MB file limit. With datasets excluded, your repo should be under 50 MB.

## After Upload

Your GitHub repository will include:
✅ Frontend (React + TypeScript)
✅ Backend (Python FastAPI)
✅ UI mockups
✅ Documentation
✅ Requirements files
✅ Base YOLO models (yolo11n.pt, yolov8n.pt - small files)

❌ NO datasets
❌ NO large model files
❌ NO training results
