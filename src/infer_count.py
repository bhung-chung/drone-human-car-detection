import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HUMAN_CLASSES = {0, 1}  # pedestrian, people
CAR_CLASSES = {3}       # car
KEEP_CLASSES = [0, 1, 3]

def draw_text(image, text, x, y, color, scale=0.6, thickness=2):
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/trained/visdrone_yolo11_human_car/weights/best.pt", help="Path to trained YOLO weights")
    parser.add_argument("--source", required=True, help="Image path or image folder")
    parser.add_argument("--out", default="outputs/counting_results")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        classes=KEEP_CLASSES,
        stream=True,
        verbose=False
    )

    for index, result in enumerate(results):
        image = result.orig_img.copy()
        boxes = result.boxes

        human_count = 0
        car_count = 0

        if boxes is not None and len(boxes) > 0:
            cls_ids = boxes.cls.int().cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy().astype(int)

            human_count = int(np.isin(cls_ids, list(HUMAN_CLASSES)).sum())
            car_count = int(np.isin(cls_ids, list(CAR_CLASSES)).sum())

            for (x1, y1, x2, y2), cls_id, score in zip(xyxy, cls_ids, confs):
                if int(cls_id) in HUMAN_CLASSES:
                    label = f"human {score:.2f}"
                    color = (0, 255, 0)
                elif int(cls_id) in CAR_CLASSES:
                    label = f"car {score:.2f}"
                    color = (255, 0, 0)
                else:
                    continue

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                draw_text(image, label, x1, max(y1 - 8, 20), color)

        summary = f"Humans: {human_count} | Cars: {car_count}"
        cv2.rectangle(image, (10, 10), (430, 52), (0, 0, 0), -1)
        draw_text(image, summary, 22, 40, (255, 255, 255), scale=0.8, thickness=2)

        if result.path:
            stem = Path(result.path).stem
        else:
            stem = f"result_{index:05d}"

        output_path = out_dir / f"{stem}_count.jpg"
        cv2.imwrite(str(output_path), image)

        print(f"Saved: {output_path} | {summary}")

if __name__ == "__main__":
    main()
