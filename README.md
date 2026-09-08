# Object detection AI

A small Python prototype that detects common objects using YOLO and OpenCV.

## Setup

Open PowerShell in this folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the pretrained detector with webcam

```powershell
python main.py
```

Press `q` to quit. The first run downloads the small `yolo11n.pt` model automatically.

To use your trained model instead:

```powershell
python main.py --model runs\detect\custom-detector\weights\best.pt
```

## Run with an image or video

```powershell
python main.py --source path\to\image.jpg
python main.py --source path\to\video.mp4 --output runs\detected.mp4
```

## Train with your own dataset

The dataset must use YOLO detection format:

```text
dataset/
	images/train/  # training images: .jpg, .png, ...
	images/val/    # validation images, used during training
	images/test/   # test images, used only for final evaluation
	labels/train/  # one .txt label per training image
	labels/val/    # one .txt label per validation image
	labels/test/   # one .txt label per test image
```

Each label line is `class_id x_center y_center width height`, with coordinates normalized from 0 to 1. For example, `0 0.5 0.5 0.2 0.3`.

Edit `dataset.yaml` so `names` contains exactly your classes. Example:

```yaml
names:
	0: bottle
	1: cup
```

Start training with:

```powershell
python train.py --data dataset.yaml --epochs 50 --batch 8 --device cpu
```

If you have an NVIDIA GPU, use `--device 0`. The best weights will be saved under `runs\detect\custom-detector\weights\best.pt`.

After training, evaluate on the test set:

```powershell
python -c "from ultralytics import YOLO; YOLO('runs/detect/custom-detector/weights/best.pt').val(data='dataset.yaml', split='test')"
```

Use roughly 70% of the images for `train`, 20% for `val`, and 10% for `test`. Do not use the same image in more than one split.

## What the pretrained version can recognize

The pre-trained model recognizes common COCO objects such as people, cars, phones, bottles, cats, and dogs. To recognize your own object category, collect and label images, then fine-tune a YOLO model on that dataset.
