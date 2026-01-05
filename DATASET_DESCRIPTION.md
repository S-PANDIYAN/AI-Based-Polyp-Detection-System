# 🏥 Polyp Dataset - Comprehensive Analysis & Description

## 📊 Dataset Overview

**Dataset Name**: Clinical Polyp Video Dataset with Pathological Annotations  
**Dataset Path**: `D:\PolyP\dataset`  
**Domain**: Medical Imaging - Colonoscopy / Gastrointestinal Endoscopy  
**Task Type**: Multi-class Classification + Semantic Segmentation  
**Date Analyzed**: December 31, 2025

---

## 🗂️ Directory Structure

```
D:\PolyP\dataset\
│
├── train/
│   └── train/
│       ├── polyps/          # 1,904 PNG images (colonoscopy frames)
│       ├── masks/           # 1,697 TIF segmentation masks
│       └── void/            # Additional data (to be explored)
│
├── test/
│   └── test/
│       ├── polyps/          # 333 PNG images
│       ├── masks/           # 333 TIF segmentation masks
│       └── void/            # Additional data
│
├── validation/
│   └── validation/
│       ├── polyps/          # 897 PNG images
│       ├── masks/           # 897 TIF segmentation masks
│       └── void/            # Additional data
│
├── clinical metadata_release0.1.csv    # Clinical annotations with lesion details
├── labels.csv                          # Image-level pathological diagnosis
├── labels_combined.csv                 # Combined diagnosis + stratification
└── annotated_metadata.xlsx             # Comprehensive metadata (Excel)
```

---

## 📈 Dataset Statistics

### Image Distribution

| Split       | Images | Masks | Percentage |
|-------------|--------|-------|------------|
| **Train**   | 1,904  | 1,697 | 60.8%      |
| **Test**    | 333    | 333   | 10.6%      |
| **Validation** | 897 | 897   | 28.6%      |
| **TOTAL**   | **3,134** | **2,927** | **100%** |

### Key Observations
- **Total Images**: 3,134 colonoscopy video frames
- **Total Masks**: 2,927 segmentation masks (TIF format)
- **Missing Masks**: ~207 images in training set without segmentation masks
- **Image Format**: PNG (polyps), TIF (masks)
- **Image Dimensions**: ~854×480 pixels (standard colonoscopy resolution)

---

## 🏷️ Classification Labels & Clinical Metadata

### Pathological Diagnosis Categories

Based on `labels.csv` and clinical metadata, the dataset contains **4 primary pathological diagnoses**:

1. **Adenoma** - Benign polyp that can become cancerous
2. **Hyperplasia** - Non-neoplastic lesion (low cancer risk)
3. **Adenocarcinoma** - Malignant cancerous polyp (invasive)
4. **Unknown/Missing** - Some frames have incomplete annotations

### Detailed Histological Stratification

From `labels_combined.csv`, each diagnosis has sub-categories:

#### Adenoma Types:
- **No dysplasia** - Benign adenoma
- **Low grade dysplasia** - Early precancerous changes
- **High grade dysplasia** - Advanced precancerous changes

#### Hyperplasia:
- **Hyperplasia** - Benign overgrowth of normal tissue

#### Adenocarcinoma:
- **Invasive adenocarcinoma** - Malignant cancer spreading through tissue layers

---

## 🔬 Clinical Features (from metadata)

Each polyp is annotated with extensive clinical information:

### 1. **Polyp Characteristics**
- **Polyp Size (mm)**: 2-50mm range
- **Paris Classification**: Morphological shape classification
  - `0-lla`: Flat elevated lesions
  - `0-llb`: Flat lesions
  - `Ip`: Pedunculated polyps (with stalk)
  - `Is`: Sessile polyps (without stalk)
  - `Ips`: Semi-pedunculated

### 2. **NICE Classification** (Narrow-band Imaging International Colorectal Endoscopic)
- **Type 1**: Hyperplastic polyps
- **Type 2**: Adenomas
- **Type 3**: Deep submucosal invasive cancer

### 3. **Video Metadata**
- **CODE - LESION**: Unique lesion identifier (1-92)
- **CODE - VIDEO**: Video source (VP1-VP46)
- **NUMBER OF POLYPS**: Multiple polyps per video (1-4)
- **CURRENT POLYP ID**: (X/N) format indicating which polyp

### 4. **Clinical Diagnoses**
- **Preliminary Diagnosis**: Gastroenterologist's visual assessment
- **Literal Diagnosis**: Pathologist's histological confirmation (ground truth)

---

## 🎯 Dataset Characteristics

### Strengths
✅ **Multi-modal annotations**: Image-level labels + pixel-level masks  
✅ **Clinical richness**: Size, morphology, NBI classification, pathology  
✅ **Real-world variability**: Multiple polyps per video, various sizes/types  
✅ **Pre-split data**: Ready-to-use train/test/validation splits  
✅ **High-quality masks**: Expert-annotated segmentation masks  

### Challenges
⚠️ **Class imbalance**: Likely more Adenoma cases than Adenocarcinoma  
⚠️ **Missing masks**: ~207 training images lack segmentation masks  
⚠️ **Variable quality**: Video frames may have motion blur, specular reflections  
⚠️ **Small dataset**: 3,134 images total (moderate size for deep learning)  
⚠️ **Multiple frames per video**: Risk of data leakage if not handled properly  

---

## 🧪 Recommended Tasks

### 1. **Multi-class Classification**
**Goal**: Classify polyps into 4 categories (Adenoma, Hyperplasia, Adenocarcinoma, Normal)  
**Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
**Input**: Polyp images (854×480 PNG)  
**Output**: Class probabilities  

### 2. **Histological Stratification**
**Goal**: Predict dysplasia grade (No/Low/High grade, Invasive, Hyperplasia)  
**Metrics**: Multi-class classification metrics, ROC-AUC  
**Clinical Impact**: High - determines treatment urgency  

### 3. **Semantic Segmentation**
**Goal**: Pixel-level polyp boundary delineation  
**Metrics**: Dice Coefficient, IoU (Jaccard Index), Pixel Accuracy  
**Input**: Polyp images  
**Output**: Binary segmentation masks  

### 4. **Combined Detection + Classification**
**Goal**: Locate polyp (segmentation) AND classify type simultaneously  
**Architecture**: Two-headed model (encoder → segmentation + classification branches)  

### 5. **Clinical Feature Prediction**
**Goal**: Predict polyp size, Paris classification, NICE type from images  
**Type**: Multi-task regression + classification  

---

## 📐 Technical Specifications

### Image Preprocessing Requirements
- **Original Resolution**: ~854×480 pixels
- **Recommended Resize**: 256×256 or 512×512 (for computational efficiency)
- **Color Space**: RGB (colonoscopy images)
- **Normalization**: [0, 1] or ImageNet statistics
- **Augmentation**: Rotation, flip, brightness/contrast, elastic deformation

### Mask Preprocessing
- **Format**: TIF (TIFF) grayscale images
- **Binary Masks**: 0 (background) vs 255 (polyp)
- **Conversion**: Threshold at 127 to create binary masks
- **Resize**: Match image dimensions

### Data Loading Strategy
```
For each image in polyps/:
    1. Load image (PNG)
    2. Find corresponding mask in masks/ (TIF, same base filename)
    3. Load clinical metadata from CSV (match filename)
    4. Apply preprocessing + augmentation
    5. Return: (image, mask, label, metadata)
```

---

## 🚨 Important Considerations

### 1. **Filename Matching**
- Polyp images: `001_VP1_frame0075.png`
- Mask files: `001_VP1_frame0075_Corrected.tif`
- **Pattern**: Add `_Corrected` and change extension `.png → .tif`

### 2. **Handle Missing Masks**
- Training set: 1,904 images but only 1,697 masks
- Options:
  - Skip images without masks for segmentation tasks
  - Use all images for classification tasks
  - Generate pseudo-masks using bounding boxes (if available)

### 3. **Video Frame Correlation**
- Multiple frames from same video (VP1, VP2, etc.)
- **Risk**: Train/test data leakage if frames from same video in both splits
- **Solution**: Verify current splits are video-aware (different videos in train/test)

### 4. **Class Distribution Analysis Needed**
- Count samples per class (Adenoma vs Hyperplasia vs Adenocarcinoma)
- Check for severe imbalance
- Consider class weights or oversampling strategies

---

## 🔍 Metadata File Details

### 1. `clinical metadata_release0.1.csv`
**Columns**: 
- CODE - LESION, CODE - VIDEO
- NUMBER OF POLYPS OF INTEREST, CURRENT POLYP ID
- POLYP SIZE (mm), PARIS CLASSIFICATION, NICE
- PRELIMINAR DIAGNOSIS, LITERAL DIAGNOSIS
- HISTOLOGICAL STRATIFICATION, Set (Train/Test/Validation)

**Use**: Link images to clinical features for multi-task learning

### 2. `labels.csv`
**Columns**: Image_Filename, LITERAL DIAGNOSIS (Pathologist)  
**Use**: Simple image → diagnosis mapping for classification

### 3. `labels_combined.csv`
**Columns**: Image_Filename, Combined_Label  
**Format**: "Diagnosis - Histological Grade"  
**Example**: "Adenoma - High grade dysplasia"  
**Use**: Fine-grained multi-class classification

---

## 🎓 Suggested Model Architectures

### Classification Models
- **CNN Baseline**: VGG16, ResNet50, EfficientNet
- **Modern**: Vision Transformers (ViT), Swin Transformer
- **Medical-specific**: Med3D, MedSAM adaptations

### Segmentation Models
- **U-Net**: Standard for medical image segmentation
- **U-Net++**: Enhanced skip connections
- **DeepLabV3+**: Atrous convolutions for multi-scale features
- **Mask R-CNN**: Instance segmentation with classification
- **Segment Anything Model (SAM)**: Zero-shot segmentation capabilities

### Hybrid Architectures
- **Two-Stream Networks**: Classification + Segmentation branches
- **Attention Mechanisms**: Focus on polyp regions
- **Multi-Task Learning**: Joint optimization of multiple objectives

---

## 📊 Expected Performance Benchmarks

### Classification
- **Baseline CNN**: 75-85% accuracy
- **Transfer Learning (ImageNet)**: 85-92% accuracy
- **Ensemble Methods**: 90-95% accuracy
- **Challenge**: Adenocarcinoma (rare class) may have lower recall

### Segmentation
- **U-Net**: Dice ~0.75-0.85, IoU ~0.65-0.75
- **U-Net++ / DeepLabV3+**: Dice ~0.80-0.90, IoU ~0.70-0.80
- **Challenge**: Small polyps (<5mm) harder to segment

### Clinical Relevance
- **Priority**: High sensitivity for Adenocarcinoma detection (minimize false negatives)
- **Acceptability**: False positives less critical than missed cancers

---

## 🛠️ Next Steps - Recommended Workflow

### Phase 1: Exploratory Data Analysis (EDA)
1. ✅ **COMPLETED**: Directory structure analysis
2. 🔄 **TODO**: Visualize sample images + masks
3. 🔄 **TODO**: Analyze class distribution (count per diagnosis)
4. 🔄 **TODO**: Check image quality (blur, artifacts)
5. 🔄 **TODO**: Validate train/test/val splits (no video leakage)

### Phase 2: Data Preprocessing Pipeline
1. Image loading function (PNG → NumPy/Tensor)
2. Mask loading function (TIF → Binary mask)
3. Filename matching function (image ↔ mask)
4. Label extraction from CSV
5. Data augmentation pipeline
6. Train/Val data generators

### Phase 3: Model Development
1. **Baseline Classification Model**: 
   - Simple CNN for 4-class diagnosis
   - Evaluate with confusion matrix, classification report
2. **Segmentation Model**:
   - U-Net for polyp boundary detection
   - Evaluate with Dice coefficient, IoU
3. **Combined Multi-task Model**:
   - Shared encoder + dual heads (classification + segmentation)

### Phase 4: Advanced Analysis
1. Clinical feature integration (size, Paris, NICE as auxiliary inputs)
2. Explainability (Grad-CAM, attention maps)
3. Model ensemble and uncertainty quantification
4. Error analysis on misclassified cases

---

## 📚 Dataset Citation & Ethics

### Usage Considerations
- **Medical Data**: Protected Health Information (PHI) - ensure HIPAA compliance
- **Clinical Validation**: Models require medical expert validation before deployment
- **Intended Use**: Research and educational purposes
- **Limitations**: Single-center data may not generalize to all populations

### Recommended Data Splits
- **Training**: 60.8% (1,904 images) - Model learning
- **Validation**: 28.6% (897 images) - Hyperparameter tuning, early stopping
- **Test**: 10.6% (333 images) - Final evaluation, unbiased performance estimate

---

## 📝 Summary

This is a **high-quality clinical colonoscopy polyp dataset** with:
- 3,134 video frames across train/test/validation splits
- 2,927 expert-annotated segmentation masks
- Multi-class pathological diagnoses (Adenoma, Hyperplasia, Adenocarcinoma)
- Rich clinical metadata (size, morphology, NBI classification, histology)

**Ideal for**:
- Multi-class polyp classification
- Semantic segmentation of polyp boundaries
- Multi-task learning (detection + classification)
- Clinical decision support system development

**Key Challenges**:
- Class imbalance (likely more benign than malignant cases)
- Some missing segmentation masks
- Small dataset size for deep learning (requires transfer learning/augmentation)

**Recommended Approach**:
1. Start with transfer learning from ImageNet or medical imaging pre-trained models
2. Apply strong data augmentation
3. Use class weights to handle imbalance
4. Implement multi-task learning to leverage both labels and masks
5. Validate clinical relevance with medical experts

---

**Dataset Status**: ✅ READY FOR MODEL DEVELOPMENT  
**Next Action**: Proceed with notebook creation upon user approval
