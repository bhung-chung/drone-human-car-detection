# Drone Human and Car Detection & Counting System

## Overview
This repository contains a computer vision pipeline designed to analyze drone/aerial images. Built using the VisDrone dataset and Ultralytics YOLO11, the system detects humans and cars, tallies a total human count, and supports video-based multi-object tracking.

## Dataset & Preprocessing
The model is trained on the VisDrone dataset. The specific Kaggle subset utilized already contained YOLO-formatted labels, eliminating the need for raw annotation conversion. Preprocessing focused on:
- Dataset verification and class balance analysis.
- Visualizing ground-truth bounding boxes.
- Handling challenges specific to aerial imagery, including small object sizes, high density, and severe occlusion.

## Model Strategy
I utilized the **YOLO11n** architecture for a balance of speed and accuracy within an 8GB VRAM constraint. 

**Critical Design Choice:** The model was trained on all 10 native VisDrone classes (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor). Training on only "humans" and "cars" would force the model to incorrectly treat visually similar vehicles (like vans or trucks) as background, degrading accuracy. 

During inference, predictions are actively filtered to evaluate only:
- Class `0` (pedestrian)
- Class `1` (people)
- Class `3` (car)

Human count is calculated as the sum of `pedestrian` and `people` detections.

## Training Configuration
- **Model:** YOLO11n (`yolo11n.pt`)
- **Epochs:** 50
- **Image Size:** 960 (to capture small aerial features)
- **Batch Size:** Auto-scaled via Ultralytics AutoBatch to maximize VRAM usage
- **Optimizations:** Mixed Precision Training (`amp=True`) enabled.

## Evaluation Metrics (Validation Set)
- **Precision:** 0.510
- **Recall:** 0.404
- **mAP50:** 0.389
- **mAP50-95:** 0.231

*(See the `/outputs/evaluation` folder for the PR curve, Confusion Matrix, and training loss charts).*

## Features
1. **Static Image Inference (`infer_count.py`):** Detects humans/cars, draws bounding boxes, and prints a live count overlay.
2. **Video Object Tracking (`track_video.py`) [Bonus]:** Implements ByteTrack to assign unique IDs to detected humans and cars across video frames, preventing double-counting of the same entity.

## How to Run

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Run Inference on Images**
```bash
python src/infer_count.py --model models/trained/visdrone_yolo11_human_car/weights/best.pt --source path/to/images --out outputs/counting_results
```

**3. Run Video Tracking**
```bash
python src/track_video.py --model models/trained/visdrone_yolo11_human_car/weights/best.pt --source path/to/video.mp4 --tracker bytetrack.yaml
```
