# SCSP-YOLO

This is the official implementation of the SCSP-YOLO model proposed in our paper, which is designed for strip surface defect detection.

## Code Structure

- `yolov5-master/`: Based on YOLOv5, integrated with SCSP module
- Training and evaluation code will be released upon paper acceptance.

## Usage

1. Clone the repo
2. Install requirements
3. Prepare your dataset (format: YOLO)
4. Train:
```bash
python train.py --data your_dataset.yaml --cfg your_model.yaml
