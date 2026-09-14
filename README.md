# RTracker & OCR

Detects sheet rolls, reads the handwritten **ply number** and **start-end** range, and draws them above the roll's box. Runs at a constant 15 fps on a Raspberry Pi 5 (2GB) with the Hailo-8L AI HAT+ (13 TOPS) and a USB camera.

```
USB cam (MJPG 1280x720 @15fps, manual exposure)
  -> YOLOv8n on Hailo-8L: roll / ply / range boxes
  -> IoU tracker (one ID per roll)
  -> text crops -> PP-OCRv4 recognizer (ONNX, CPU thread, digits . - only)
  -> vote over several frames -> overlay + annotated MP4 + CSV + console
```

| File | Purpose |
|---|---|
| `rtracker_ocr.py` | Main app |
| `detector.py` | Hailo (`.hef`) / Ultralytics (`.pt`) detector |
| `ocr.py` | Text recognizer + format check (`99`, `99.45-109.71`) |
| `extract_frames.py` | Frames from `videos/` for labeling and Hailo calibration |
| `train.py` | Train YOLOv8n, export `models/rolls.pt` + `models/rolls.onnx` |
| `models/ppocr_rec.onnx` | PaddleOCR PP-OCRv4 recognition model |

## 1. Label (PC)
```
pip install -r requirements.txt
python extract_frames.py --every-sec 1.0        # -> dataset/images
```
Label in YOLO format (CVAT, Roboflow or labelImg) and save the `.txt` files to `dataset/labels/`. Use 3 classes in this order:
- `0 roll`: the whole roll
- `1 ply`: the tight box around the ply number
- `2 range`: the tight box around the start-end text

Keep the text boxes tight, because the crop goes directly to the OCR. Label frames where the roll is lying, standing, in a hand and on the machine.

## 2. Train (PC)
```
python train.py --epochs 100
```

## 3. Compile for Hailo-8L (Ubuntu x86 or WSL2, with the Hailo AI Software Suite / Model Zoo)
```
hailomz compile yolov8n --ckpt models/rolls.onnx --hw-arch hailo8l --calib-path dataset/images --classes 3
mv yolov8n.hef models/rolls.hef
```
The compiler version must match the HailoRT version on the Pi (`hailortcli fw-control identify`).

## 4. Run on the Pi 5
```
sudo apt install hailo-all python3-opencv v4l-utils
python3 -m venv --system-site-packages venv && . venv/bin/activate
pip install onnxruntime
python3 rtracker_ocr.py --source /dev/video0 --model models/rolls.hef --exposure 100
```
- `--exposure`: manual exposure in 100 µs units. At 15 fps the max is 666. Use about 50–150 so moving rolls don't blur, and lower it if the white fabric saturates. Leave it out for auto exposure. The app turns off `exposure_dynamic_framerate` so auto exposure can't drop the frame rate below 15 fps.
- `--votes 3`: how many matching OCR reads are needed before a value is confirmed. The box turns **green** when both values are confirmed and **orange** while it is still reading.
- `--no-show`: headless. Press `q` to quit the window.
- Outputs: `output/annotated.mp4` (15 fps) and `output/readings.csv` (`timestamp, track_id, ply, start, end`). Each confirmed roll is also printed to the console.

## Quick try without training (PC)
```
pip install opencv-python numpy onnxruntime
python rtracker_ocr.py --source videos/cam2_test1.mp4 --model models/ppocr_det.onnx
```
This mode uses PaddleOCR's text detector (`models/ppocr_det.onnx`) in place of YOLO. It finds any text in the frame, reads it, and shows only text that matches the ply or start-end format. The box drawn is around the text, not the whole roll. Works well on clear marker writing (`cam2_test*`, `cam0_test*`), but it misses faint or thin writing and runs at about 12–15 fps on a PC CPU. It is for trying things out only, not for the Pi.

## Test on a PC with the recorded videos (trained model)
```
python rtracker_ocr.py --source videos/cam0_onsite.mp4 --model models/rolls.pt
```
Video files are resampled to 15 fps and played in real time.

## Notes
- **OCR accuracy:** clear marker writing (for example the red marker in `cam2_test*`) reads well (`54.7-9.21` at 0.93 confidence). The thin pen writing in `cam0_onsite*` is only about 20 px tall, and the pretrained recognizer often misreads it. The best fixes are:
  1. thicker, darker marker
  2. higher resolution or the camera closer to the roll
  3. fine-tuning the PP-OCR recognizer on crops from this site
- A short dash that looks like a dot is corrected automatically: `31.3.34.1` becomes `31.3-34.1`.
- **Inverted or sideways text:** the ply number is always written above the start-end, so the position of the ply box relative to the range box shows which way is up. Ply below range means the text is rotated 180°, and ply left or right of range means it is rotated 90°. Both crops are turned upright before OCR. Digits like 0, 1, 2, 5, 8 and 6↔9 look valid either way up, so OCR scores alone can't tell. If only one text line is visible, the OCR tries both orientations and keeps the better valid reading. Label inverted and standing rolls with the same classes; `train.py` uses up-down and left-right flips.
