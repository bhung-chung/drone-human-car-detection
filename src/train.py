import argparse
from ultralytics import YOLO

def parse_batch(value: str):
    """Safely parse the batch argument to handle -1 for AutoBatch."""
    value = float(value)
    return int(value) if value.is_integer() else value

def main():
    parser = argparse.ArgumentParser()
    # Default points to the config file we created earlier
    parser.add_argument("--data", default="configs/visdrone.yaml", help="Path to dataset YAML")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model: yolo11n.pt, yolo11s.pt, etc.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=parse_batch, default=-1)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", default="visdrone_yolo11_human_car")
    args = parser.parse_args()

    # Load the base YOLO model
    model = YOLO(args.model)

    # Start the training process
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,      # -1 uses Ultralytics AutoBatch to maximize your VRAM usage safely
        device=args.device,
        workers=args.workers,
        amp=True,              # Mixed Precision Training (crucial for saving memory)
        cache=False,           # Set to False to save RAM
        patience=15,           # Stops training early if the model stops improving for 15 epochs
        project="models/trained",  # Where the weights and charts will be saved
        name=args.name,
        exist_ok=True,
        plots=True             # Automatically generates evaluation charts for your README
    )

if __name__ == "__main__":
    main()
