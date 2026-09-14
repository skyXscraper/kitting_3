"""Compile models/rolls.onnx (YOLOv8n-OBB from train.py) to models/rolls.hef for the Hailo-8L.
Run on x86 Linux or WSL2 with the Hailo Dataflow Compiler (hailo_sdk_client) installed:
    python compile_hailo.py
The network is cut at the 9 raw head convolutions (box DFL 64ch, class 3ch, angle 1ch at 3 scales);
box/angle decoding and rotated NMS run on the Pi in detector.HailoDetector. The DFC version must match
the HailoRT version on the Pi."""
import glob

import cv2
import numpy as np
from hailo_sdk_client import ClientRunner

from detector import letterbox

END_NODES = [f"/model.22/{b}.{i}/{b}.{i}.2/Conv" for i in range(3) for b in ("cv2", "cv3", "cv4")]

# calibration images = training frames prepared exactly like on the Pi (letterbox 640, RGB, 0-255)
calib = np.stack([letterbox(cv2.imread(p), 640, 640)[0][:, :, ::-1]
                  for p in sorted(glob.glob("dataset/train/images/*.jpg"))]).astype(np.float32)

runner = ClientRunner(hw_arch="hailo8l")
runner.translate_onnx_model("models/rolls.onnx", "rolls", start_node_names=["images"], end_node_names=END_NODES,
                            net_input_shapes={"images": [1, 3, 640, 640]})
runner.load_model_script("normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n")
runner.optimize(calib)
with open("models/rolls.hef", "wb") as f:
    f.write(runner.compile())
print("Saved models/rolls.hef")
