import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/trained/visdrone_yolo11_human_car/weights/best.pt", help="Path to trained YOLO weights")
    parser.add_argument("--source", required=True, help="Path to the input video file")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    model = YOLO(args.model)

    # Run tracking, filtering specifically for pedestrian (0), people (1), and car (3)
    model.track(
        source=args.source,
        tracker=args.tracker,
        conf=args.conf,
        imgsz=args.imgsz,
        classes=[0, 1, 3], 
        save=True,
        project="outputs",
        name="tracking_results",
        exist_ok=True
    )

if __name__ == "__main__":
    main()
