from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def draw_detections(frame, results) -> None:
    annotated = results[0].plot()
    frame[:] = annotated


def run(source: int | str, confidence: float, output: str | None, model_path: str) -> None:
    model = YOLO(model_path)
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    writer = None
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter.fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            results = model.predict(frame, conf=confidence, verbose=False)
            draw_detections(frame, results)

            if writer:
                writer.write(frame)

            cv2.imshow("Object detection - press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect objects from a webcam, image, or video.")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (default: 0), image path, or video path.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.45,
        help="Minimum confidence from 0 to 1 (default: 0.45).",
    )
    parser.add_argument(
        "--output",
        help="Optional output video path, for example runs/detected.mp4.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Model weights to use (default: yolo11n.pt).",
    )
    args = parser.parse_args()

    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be greater than 0 and at most 1")

    run(parse_source(args.source), args.confidence, args.output, args.model)


if __name__ == "__main__":
    main()
