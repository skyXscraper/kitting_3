"""Save frames from the videos for labeling (and as Hailo calibration images).
python extract_frames.py --every-sec 1.0
"""
import argparse
import glob
import os

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--videos", default="videos")
ap.add_argument("--out", default="dataset/images")
ap.add_argument("--every-sec", type=float, default=1.0)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
total = 0
for path in sorted(glob.glob(os.path.join(args.videos, "*.mp4"))):
    cap = cv2.VideoCapture(path)
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 15) * args.every_sec))
    name, i = os.path.splitext(os.path.basename(path))[0], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            cv2.imwrite(os.path.join(args.out, f"{name}_{i:05d}.jpg"), frame)
            total += 1
        i += 1
    print(path, "done")
print(f"{total} frames saved to {args.out}")
