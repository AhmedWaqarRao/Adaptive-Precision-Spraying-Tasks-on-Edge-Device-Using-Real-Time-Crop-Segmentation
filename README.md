## 📌 Project Title
# 🚀 Adaptive-Precision-Spraying-Tasks-on-Edge-Device-Using-Real-Time-Crop-Segmentation

"Adaptive Edge AI Deployment for Real-Time Crop Segmentation Using Scenario-Aware Dynamic Model Pruning"

## 🔍 Overview

This repository presents an advanced and deployable research implementation focused on real-time crop segmentation using deep neural networks, optimized for edge computing environments. It is designed for application in smart agriculture, where power efficiency, low-latency inference, and compute-constrained hardware make conventional deep models impractical.

The work emphasizes:

* Edge AI deployment using the YOLOv8L-Seg model
* Entropy-guided dynamic pruning (F2Zip) for real-time adaptive model compression
* Acceleration with TensorRT for high-throughput, low-latency inference
* AIoT-compatible inference pipelines for on-field crop detection

The solution excludes any spraying or actuation components and focuses purely on the deep learning, pruning, optimization, and deployment aspects of the pipeline.

---

## 🎯 Objectives

* Achieve accurate segmentation of tobacco crops from aerial RGB images
* Dynamically compress the model based on image scene complexity without retraining
* Deploy optimized models on Jetson AGX Xavier using TensorRT and ONNX
* Demonstrate real-time performance on video and image inference tasks

---

## 🧠 Key Components

### 🔹 Deep Neural Network: YOLOv8L-Seg

* High-resolution segmentation of crop boundaries
* Supports both bounding box and pixel-wise mask prediction
* Trained on custom-labeled datasets (TobSet and Tobacco Aerial Dataset)

### 🔹 Dynamic Scenario-Aware Pruning (F2Zip)

This pruning technique compresses the model without fine-tuning, making it ideal for edge deployment:

* Computes entropy of scenes to estimate visual complexity
* Evaluates channel importance using both:

  * Static metrics (L1 norm of convolutional filters)
  * Dynamic metrics (activation variance across batches)
* Uses a multi-objective knapsack algorithm to select which channels to prune
* Prunes only convolutional layers; attention and bottlenecks remain untouched

### 🔹 Deployment and Inference

* Compressed model exported to ONNX, optimized via TensorRT, and deployed to Jetson Xavier AGX 32GB
* Inference interfaces developed for:

  * Static image segmentation
  * Real-time video inference

---

## 📊 Performance Metrics

| Model Variant          | Precision | Recall | mAP\@0.5 | mAP\@0.5:0.95 | Inference Time |
| ---------------------- | --------- | ------ | -------- | ------------- | -------------- |
| YOLOv8L-Seg (Original) | 0.976     | 0.992  | 0.992    | 0.955         | 1034 ms        |
| Pruned YOLOv8L-Seg     | 0.950     | 0.760  | 0.890    | 0.790         | 60 ms          |
| TensorRT Optimized     | 0.940     | 0.830  | 0.900    | 0.810         | 10.4 ms        |

---

## 🧪 Tools and Code Modules

### 🧾 Image Inference

```python
from ultralytics import YOLO
model = YOLO("best.pt")
results = model("image.jpg")
```

### 🎥 Video Inference

```python
cap = cv.VideoCapture("input.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)
    out.write(results[0].plot())
```

---

## 📂 Repository Structure

```
├── yolov8-seg-training.ipynb       # Full pipeline for training and export
├── segmentation_inference.py       # Image-level inference demo
├── video_inference.py              # Video segmentation
├── yolov8l-seg-pruned.engine       # TensorRT model for edge
├── yolov8l-seg-pruned-deep.pt      # Pruned YOLOv8 model
├── F2zip_(3).ipynb                 # Entropy + pruning code
├── ALGO_1.txt / Algo 2.txt         # Core pruning algorithms
├── output_video.mp4                # Sample output video
├── YOLO Deployment on Edge.pdf     # Deployment documentation
├── Choosing YOLO for Jetson.docx   # Comparative study of YOLO variants
├── Code Explanation.docx           # Function-level pruning analysis
```

---

## 📚 Theoretical Foundation

* Entropy Estimation: Measures image complexity as a proxy for pruning aggressiveness
* Importance Score Calculation: Combines L1-norm and activation-based ranking
* Multi-constraint Knapsack: Selects channels to prune while satisfying latency, FLOPs, and parameter constraints
* Based on F2Zip pruning algorithm

---

## 🧾 Academic Integration

This repository serves as the implementation base for the author’s thesis:
"Dynamic Scenario-Aware Model Pruning for Edge-AI Deployment"

* Focused on Edge AI + Deep Learning Compression + AIoT Inference
* Application in real-time, rural agriculture environments
* Delivered with hardware-aware pruning without retraining overhead

---

Specialization:

* Edge AI and AIoT Systems
* Deep Learning Compression
* Embedded Deep Learning Deployment

---

## 🌿 Conclusion

This project bridges advanced pruning theory with real-world deployment needs in smart agriculture. It demonstrates that deep learning models can be intelligently compressed and deployed on edge devices without significant accuracy loss, enabling scalable, low-latency AI for the field. Future improvements include integrating thermal/multispectral sensors and federated learning for cross-location model updating.
