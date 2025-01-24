# Autonomous Driving - Car Detection

## Overview
This project implements object detection using the powerful YOLO (You Only Look Once) model. The implementation focuses on car detection for autonomous driving scenarios, leveraging pre-trained YOLO weights for efficiency. The project involves building, fine-tuning, and evaluating YOLO-based object detection on a car dataset.

## Key Features
- Detect objects in a car detection dataset using the YOLO model.
- Implement non-max suppression (NMS) to improve detection accuracy.
- Calculate Intersection over Union (IoU) for bounding box evaluation.
- Process and visualize bounding boxes for detected objects.

## Dataset
- **Source**: [Drive.ai Sample Dataset](http://creativecommons.org/licenses/by/4.0/)
- The dataset consists of images captured using a car-mounted camera, with labeled bounding boxes for cars and other objects.

## Requirements
- **Python Libraries**:
  - `TensorFlow`
  - `Keras`
  - `NumPy`
  - `Matplotlib`
  - `Pillow`
- Install dependencies:
  ```bash
  pip install tensorflow keras numpy matplotlib pillow
## Workflow

### 1. Model Details
- **Input**: Images resized to (608, 608, 3).
- **Output**: Bounding boxes for detected objects, represented as (19, 19, 5, 85) tensors.
- **Pre-trained YOLO weights**: Used for model initialization.

### 2. Implementations
- **Thresholding**:
  - Filter boxes based on class scores.
- **Non-Max Suppression**:
  - Eliminate overlapping bounding boxes with IoU-based filtering.
- **Bounding Box Visualization**:
  - Display bounding boxes and object class probabilities on images.

### 3. Evaluation
- Ran YOLO on sample images to detect and localize objects.
- Output includes bounding boxes, confidence scores, and detected class labels.

---

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
2. Place the dataset in the `images/` directory.
3. Run the Jupyter Notebook or Python script to detect objects:
   ```bash
   python yolo_detection.py
## View Results
- Results with bounding boxes are saved in the `out/` directory.

---

## Results

### Detected Objects
- Multiple cars, buses, traffic lights, and other objects detected with bounding boxes.

### Metrics
- High detection accuracy achieved using IoU and non-max suppression techniques.

---

## Discussion

### Strengths
- Efficient object detection with high accuracy.
- Visualizations provide clear insights into YOLO's predictions.

### Challenges
- Computationally intensive, especially for large datasets or real-time applications.
- Requires a well-labeled dataset for optimal performance.

---

## Future Improvements
- Fine-tune YOLO on larger datasets for better generalization.
- Experiment with YOLOv3 or YOLOv4 for improved accuracy and speed.
- Deploy the model on edge devices for real-time detection in autonomous vehicles.

---

## References
- **You Only Look Once: Unified, Real-Time Object Detection**
- **YOLO9000: Better, Faster, Stronger**
- **YAD2K: Yet Another Darknet 2 Keras**
- **Drive.ai Sample Dataset**

