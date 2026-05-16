# Drone Human and Car Detection & Counting System

## Overview

This project builds a computer vision pipeline for drone and aerial images using the VisDrone dataset and Ultralytics YOLO11. The system detects humans and cars, counts total humans in each image, visualizes bounding boxes, and includes optional video tracking.

Main features:

* Human and car detection in aerial images
* Human counting using bounding box detections
* Output visualization with labels and count overlay
* YOLO11-based model training
* Optional video tracking with ByteTrack

---

## Dataset

Dataset: VisDrone Kaggle dataset

```text
https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset
```

The Kaggle version already includes YOLO-format labels, so no raw annotation conversion was required.

VisDrone classes:

```text
0: pedestrian
1: people
2: bicycle
3: car
4: van
5: truck
6: tricycle
7: awning-tricycle
8: bus
9: motor
```

For this assessment, the final output uses:

```text
0: pedestrian
1: people
3: car
```

Human count is calculated as:

```text
human count = pedestrian detections + people detections
```

---

## Dataset Setup

Place the dataset in this structure:

```text
data/VisDrone/VisDrone_Dataset/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── labels/
├── VisDrone2019-DET-val/
│   ├── images/
│   └── labels/
└── VisDrone2019-DET-test-dev/
    └── images/
```

Recommended `configs/visdrone.yaml`:

```yaml
path: data/VisDrone/VisDrone_Dataset

train: VisDrone2019-DET-train/images
val: VisDrone2019-DET-val/images
test: VisDrone2019-DET-test-dev/images

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
```

---

## Repository Structure

```text
configs/
  visdrone.yaml

notebooks/
  dataset_exploration.ipynb

src/
  run_exploration.py
  train.py
  infer_count.py
  track_video.py

outputs/
  charts/
  counting_results/
  evaluation/
```

---

## Dataset Exploration

Since the dataset already has YOLO labels, preprocessing focused on verification and visualization:

* Checked image and label counts
* Analyzed class distribution
* Visualized ground-truth bounding boxes
* Identified aerial image challenges such as small objects, occlusion, density, and class imbalance

Run:

```bash
python src/run_exploration.py
```

Generated outputs:

```text
outputs/charts/class_distribution.png
outputs/charts/ground_truth_visualization.png
```

---

## Model Strategy

The model was trained on all 10 VisDrone classes, not only humans and cars. This helps because visually similar classes such as car, van, truck, bus, and motor should not be treated as background during training.

During inference, predictions are filtered to only:

```text
pedestrian, people, car
```

This keeps the final output focused on the assessment requirement while preserving better training quality.

---

## Training Configuration

```text
Model: YOLO11n
Epochs: 50
Image size: 960
Batch size: AutoBatch
Mixed precision: Enabled
Training classes: All 10 VisDrone classes
Inference classes: pedestrian, people, car
```

Image size `960` was used to better detect small aerial objects. AutoBatch and mixed precision were used to reduce GPU memory issues.

---

## Installation

```bash
pip install -r requirements.txt
```

If `requirements.txt` is unavailable:

```bash
pip install ultralytics opencv-python matplotlib pandas numpy Pillow tqdm PyYAML
```

---

## Training

```bash
python src/train.py \
  --data configs/visdrone.yaml \
  --model yolo11n.pt \
  --epochs 50 \
  --imgsz 960 \
  --batch -1
```

For lower VRAM:

```bash
python src/train.py \
  --data configs/visdrone.yaml \
  --model yolo11n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 4
```

Best model path:

```text
models/trained/visdrone_yolo11_human_car/weights/best.pt
```

---

## Validation

```bash
yolo detect val \
  model=models/trained/visdrone_yolo11_human_car/weights/best.pt \
  data=configs/visdrone.yaml \
  imgsz=960
```

Validation results:

```text
Precision: 0.510
Recall: 0.404
mAP50: 0.389
mAP50-95: 0.231
```

Evaluation plots are saved in:

```text
outputs/evaluation/
```

---

## Image Inference and Counting

Run inference on images:

```bash
python src/infer_count.py \
  --model models/trained/visdrone_yolo11_human_car/weights/best.pt \
  --source path/to/images \
  --out outputs/counting_results \
  --conf 0.25 \
  --imgsz 960
```

The script keeps only classes `[0, 1, 3]`, draws bounding boxes, and displays:

```text
Humans: pedestrian + people
Cars: car
```

Outputs are saved in:

```text
outputs/counting_results/
```

---

## Optional Video Tracking

Run tracking with ByteTrack:

```bash
python src/track_video.py \
  --model models/trained/visdrone_yolo11_human_car/weights/best.pt \
  --source path/to/video.mp4 \
  --tracker bytetrack.yaml \
  --conf 0.25 \
  --imgsz 960
```

For BoT-SORT:

```bash
python src/track_video.py \
  --model models/trained/visdrone_yolo11_human_car/weights/best.pt \
  --source path/to/video.mp4 \
  --tracker botsort.yaml
```

The tracking script assigns tracking IDs to humans and cars across frames. The current version provides ID-based visualization and can be extended to full unique object counting across a video.

Tracking outputs are saved in:

```text
outputs/tracking_results/
```
> Note: The generated tracking video is not included in this GitHub repository because of file size limits. It is provided separately in the submitted Google Drive folder.

---

## Results

The repository includes:

* Class distribution chart
* Ground-truth visualization
* Sample detection and counting outputs
* Evaluation curves and confusion matrix

Important folders:

```text
outputs/charts/
outputs/counting_results/
outputs/evaluation/
```

---

## Strengths

* Uses a drone-specific dataset
* Trains on all native VisDrone classes
* Filters only humans and cars during inference
* Counts humans using both pedestrian and people classes
* Includes detection, counting, visualization, and optional tracking
* Uses memory-friendly settings such as AutoBatch and mixed precision

---

## Limitations

* Very small or occluded humans may be missed
* Similar vehicle classes can be confused
* Counting is image-based in the inference script
* Full unique video counting is not implemented yet
* Higher image size improves detection but requires more GPU memory

---

## Future Improvements

* Train a larger model such as YOLO11s or YOLO11m
* Add unique ID-based counting for video clips
* Tune confidence thresholds separately for humans and cars
* Add FPS benchmarking
* Improve small-object detection using tiled inference

---

## Trained Weights

The trained `.pt` model file is not included in the repository because of file size.

Expected path after downloading or training:

```text
models/trained/visdrone_yolo11_human_car/weights/best.pt
```
The trained model weights and full tracking video are provided separately in the submitted Google Drive folder because of file size limits.

---

## Author

Developed for the Antlings Internship Program technical assessment.
