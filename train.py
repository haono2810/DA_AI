from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a custom YOLO object detector.")
    parser.add_argument("--data", default="dataset.yaml", help="Path to the dataset YAML file.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base weights for transfer learning.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size; lower it if memory is limited.")
    parser.add_argument("--device", default="cpu", help="cpu, 0 for the first GPU, or auto.")
    parser.add_argument("--name", default="custom-detector", help="Name of the training run.")
    args = parser.parse_args()

    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0:
        parser.error("--epochs, --imgsz, and --batch must be positive")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=15,
        plots=True,
    )


if __name__ == "__main__":
    main()