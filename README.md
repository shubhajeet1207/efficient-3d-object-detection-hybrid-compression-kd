# 🚀 Efficient 3D Object Detection via Hybrid Compression & Knowledge Distillation

Accelerate and compress 3D object detection models by combining **category-aware pruning** with **knowledge distillation** — implemented as an extension to the OpenPCDet framework.

---

## 📖 Overview

This project introduces the `KDPruningDetector`, a novel detector that trains a compact **student** network to mimic a larger **teacher** while **dynamically pruning channels** based on object-category importance.

### ✨ Key Contributions

- **Hybrid Model Compression**  
  Smart, category-aware channel pruning that preserves essential features per object category.

- **Feature-level Distillation**  
  Lightweight projection networks that align student features with the teacher’s.

- **Label-level Distillation**  
  Soft target supervision to guide classification and regression outputs.

- **End-to-End Training**  
  Joint optimization of supervised and distillation objectives for maximum efficiency.

---

## 📂 Repository Structure

```plaintext
OpenPCDet/
├── pcdet/
│   ├── models/
│   │   ├── detectors/
│   │   │   ├── __init__.py
│   │   │   └── kd_pruning_detector.py      # ← New detector implementation
│   │   ├── backbones_3d/
│   │   ├── dense_heads/
│   │   ├── roi_heads/
│   │   └── model_utils/
│   ├── utils/
│   └── datasets/
├── tools/
│   ├── cfgs/
│   │   └── kitti_models/
│   │       └── kd_pruning.yaml             # ← New config file
├── docs/
└── data/
