"""Train YOLOv8n-OBB on the Roboflow dataset and export ONNX for the Hailo compiler.
Uses the dataset's own split: dataset/data.yaml (train/valid/test, classes: 0=ply 1=range 2=roll)
python train.py --epochs 100
"""
import argparse
import os
import shutil

from ultralytics import YOLO

if __name__ == "__main__":  # required on Windows: data-loader workers re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    model = YOLO("yolov8n-obb.pt")
    # up-down + left-right flips so the detector also finds ply/range text on inverted (180 deg) rolls
    model.train(data=os.path.abspath("dataset/data.yaml"), epochs=args.epochs, imgsz=640, batch=args.batch,
                fliplr=0.5, flipud=0.5, project="runs", name="rolls", exist_ok=True)

    best = "runs/rolls/weights/best.pt"
    os.makedirs("models", exist_ok=True)
    shutil.copy(best, "models/rolls.pt")
    onnx = YOLO(best).export(format="onnx", imgsz=640, opset=11)
    shutil.copy(onnx, "models/rolls.onnx")
    print("Saved models/rolls.pt and models/rolls.onnx")
