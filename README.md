# MURA-RL-DeepEnsembleV2-82.4

## Overview  
MURA-RL-DeepEnsembleV2-82.4 is a reinforcement learning-enhanced ensemble model for musculoskeletal X-ray classification, inspired by **DeepSeek-R1**'s RL-driven reasoning framework. The model dynamically adjusts its prediction weighting using a reinforcement learning agent, optimizing classification across body parts.  

Two model versions are included:  
- **ultimate_ensemble_RAAAH.pth** (82.4% Val Acc, 0.65 Kappa) – Downloadable from **Releases** or at this link: https://github.com/blahmin/MURA-RL-DeepEnsembleV2-82.4/releases/tag/model  
- **enhanced_ensemble_best.pth** – Pretrained model used as a base for RL fine-tuning.  

## Model Performance  
- **Validation Accuracy:** 82.4%  
- **Cohen’s Kappa Score:** 0.65  

## Model Architecture  
### **1. BaseModel (Residual CNN with Attention)**
- Uses **EnhancedBlock** for residual learning and feature extraction.  
- **Attention mechanism** to focus on critical spatial features.  
- **Adaptive average pooling** for robust generalization.  

### **2. RL-Inspired Weighting Agent**
- An RL-based Softmax agent dynamically learns **model weighting per body part**.  
- Weights are **self-adjusting**, based on reward signals.  

### **3. Specialized Attention for Body Parts**
- Body-part-specific attention layers optimize predictions for **SHOULDER** and **ELBOW**.  
- Other body parts use a **default RL-optimized weighting strategy**.  

## Training Pipeline  
- **Dataset:** MURA (Musculoskeletal Radiographs)  
- **Augmentations:** Random rotations, flips, color jitter, affine transformations.  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** AdamW  
- **Scheduler:** OneCycleLR  
- **Batch Size:** 64  

## Installation  
Install dependencies:  
pip install torch torchvision albumentations pillow

python
Copy
Edit

## Download the RL Model
The RL-trained model ultimate_ensemble_RAAAH.pth is too large for GitHub version control and is available in GitHub Releases @ https://github.com/blahmin/MURA-RL-DeepEnsembleV2-82.4/releases/tag/model  

## Future Improvements
Improve reward modeling for RL agent fine-tuning.
Test misclassification-aware retraining using reinforcement loss.
Implement Grad-CAM heatmaps to improve model interpretability.
This model explores RL-powered medical imaging classification, with future iterations focusing on accuracy, efficiency, and interpretability.
Outperform radiologists.
