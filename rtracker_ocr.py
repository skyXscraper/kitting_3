"""Roll tracking + OCR of ply number and start-end range at a constant 15 fps.

Pi 5 (USB camera):  python3 rtracker_ocr.py --source /dev/video0 --model models/rolls.hef --exposure 100
PC (video file):    python rtracker_ocr.py --source videos/cam0_onsite.mp4 --model models/rolls.pt
"""
import argparse
import csv
import os
import queue
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime

import cv2
import numpy as np

from detector import load_detector, rect_bounds, rect_points, PLY, RANGE, ROLL, TEXT
from ocr import TextReader, PLY_RE, RANGE_RE, align_rect, crop_rect, text_rotation

FPS = 15
OCR_EVERY = 5      # frames between OCR attempts per text field of a roll
MIN_SCORE = 0.5    # minimum OCR confidence for a reading to count as a vote
MAX_MISSED = FPS   # drop a track after 1 s without a detection


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)


class Track:
    next_id = 1

    def __init__(self, rect):
        self.id, Track.next_id = Track.next_id, Track.next_id + 1
        self.rect, self.missed, self.logged = rect, 0, False  # rotated rect (cx, cy, w, h, theta)
        self.votes = {"ply": Counter(), "range": Counter()}
        self.last_ocr = {"ply": -OCR_EVERY, "range": -OCR_EVERY, "auto": -OCR_EVERY}

    def value(self, field, min_votes):
        if not self.votes[field]:
            return None, False
        text, n = self.votes[field].most_common(1)[0]
        return text, n >= min_votes


def open_source(src, w, h, exposure):
    if src.isdigit() or src.startswith("/dev/video"):
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src, cv2.CAP_V4L2 if sys.platform == "linux" else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # MJPG needed for 720p@15 over USB
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if sys.platform == "linux":
            dev = src if src.startswith("/dev/") else f"/dev/video{src}"
            # stop auto-exposure from lowering the frame rate in dim light
            subprocess.run(["v4l2-ctl", "-d", dev, "-c", "exposure_dynamic_framerate=0"], capture_output=True)
        if exposure:  # V4L2 manual exposure, units of 100 us (max 666 at 15 fps); short = less motion blur
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        return cap, False
    return cv2.VideoCapture(src), True


def frames_15fps(cap, is_file):
    """Yield frames at 15 fps: resample video files, pace playback to real time."""
    step = (cap.get(cv2.CAP_PROP_FPS) or FPS) / FPS if is_file else 1.0
    i, next_i, t_next = 0, 0.0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        i += 1
        if i - 1 < next_i:
            continue
        next_i += step
        time.sleep(max(0.0, t_next - time.time()))
        t_next = max(t_next + 1.0 / FPS, time.time() - 1.0 / FPS)
        yield frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/dev/video0", help="camera index, /dev/videoN or video file")
    ap.add_argument("--model", default="models/rolls.hef", help=".hef on the Pi, .pt on a PC")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--exposure", type=int, default=None, help="manual exposure (100 us units), omit for auto")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--votes", type=int, default=3, help="matching OCR reads needed to confirm a value")
    ap.add_argument("--out", default="output/annotated.mp4")
    ap.add_argument("--csv", default="output/readings.csv")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    detector = load_detector(args.model, args.conf)
    reader = TextReader()
    cap, is_file = open_source(args.source, args.width, args.height, args.exposure)
    if not cap.isOpened():
        sys.exit(f"Cannot open source {args.source}")

    new_csv = not os.path.exists(args.csv)
    csv_file = open(args.csv, "a", newline="")
    log = csv.writer(csv_file)
    if new_csv:
        log.writerow(["timestamp", "track_id", "ply", "start", "end"])

    # OCR runs in a worker thread so detection/display stay at 15 fps; extra jobs are dropped
    jobs = queue.Queue(maxsize=8)
    text_only = getattr(detector, "text_only", False)

    def ocr_worker():
        while True:
            track, field, crop, rotation = jobs.get()
            # "auto" = unclassified text line (no-training mode): keep whichever format it matches
            for f in (["range", "ply"] if field == "auto" else [field]):
                text, score = reader.read(crop, PLY_RE if f == "ply" else RANGE_RE, rotation)
                if text and score >= MIN_SCORE:
                    track.votes[f][text] += 1
                    break

    threading.Thread(target=ocr_worker, daemon=True).start()

    tracks, logged_rolls, writer, frame_no, t_fps, fps = [], set(), None, 0, time.time(), 0.0
    for frame in frames_15fps(cap, is_file):
        frame_no += 1
        fh, fw = frame.shape[:2]
        dets = detector.detect(frame)
        rolls = [d[2] for d in dets if d[0] == ROLL]
        texts = [d for d in dets if d[0] in (PLY, RANGE, TEXT)]

        # --- track rolls (greedy IoU matching on the rotated boxes' outer bounds) ---
        unmatched = list(range(len(rolls)))
        for t in tracks:
            best = max(unmatched, key=lambda j: iou(rect_bounds(t.rect), rect_bounds(rolls[j])), default=None)
            if best is not None and iou(rect_bounds(t.rect), rect_bounds(rolls[best])) > 0.3:
                t.rect, t.missed = rolls[best], 0
                unmatched.remove(best)
            else:
                t.missed += 1
        tracks = [t for t in tracks if t.missed <= MAX_MISSED] + [Track(rolls[j]) for j in unmatched]

        # --- assign text boxes to the roll whose rotated box contains their centre ---
        roll_texts = {}
        for d in texts:
            for t in tracks:
                if t.missed == 0 and cv2.pointPolygonTest(rect_points(t.rect), d[2][:2], False) >= 0:
                    roll_texts.setdefault(t, []).append(d)
                    break

        # --- text orientation from layout (ply above start-end), then send straightened crops to OCR ---
        for t, ds in roll_texts.items():
            rng = [d[2] for d in ds if d[0] == RANGE]
            if rng:  # point every text box of this roll the same way as the range box
                ds = [(c, s, align_rect(r, rng[0][4])) for c, s, r in ds]
            ply = [d[2] for d in ds if d[0] == PLY]
            rotation = text_rotation(ply[0], rng[0]) if ply and rng else None  # None: OCR tries both ways
            for cls, _, rect in ds:
                field = {PLY: "ply", RANGE: "range", TEXT: "auto"}[cls]
                confirmed = t.value("ply", args.votes)[1] and t.value("range", args.votes)[1] \
                    if field == "auto" else t.value(field, args.votes)[1]
                due = frame_no - t.last_ocr[field] >= OCR_EVERY or t.last_ocr[field] == frame_no
                if not confirmed and due:
                    try:
                        jobs.put_nowait((t, field, crop_rect(frame, rect), rotation))
                        t.last_ocr[field] = frame_no
                    except queue.Full:
                        pass
                if not text_only:
                    cv2.polylines(frame, [np.int32(rect_points(rect))], True, (255, 200, 0), 1)

        # --- draw results, log confirmed readings ---
        for t in tracks:
            if t.missed or (text_only and not any(t.votes.values())):  # text-only: hide text that isn't a reading
                continue
            ply, ply_ok = t.value("ply", args.votes)
            rng, rng_ok = t.value("range", args.votes)
            done = ply_ok and rng_ok
            color = (0, 200, 0) if done else (0, 165, 255)
            x1, y1 = map(int, rect_bounds(t.rect)[:2])
            label = f"#{t.id} Ply {ply or '?'} | {rng or '?'}"
            cv2.polylines(frame, [np.int32(rect_points(t.rect))], True, color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            ty = max(th + 8, y1)
            cv2.rectangle(frame, (x1, ty - th - 8), (x1 + tw + 6, ty), color, -1)
            cv2.putText(frame, label, (x1 + 3, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            if done and not t.logged:
                t.logged = True
                if (ply, rng) not in logged_rolls:  # same roll seen again -> don't log twice
                    logged_rolls.add((ply, rng))
                    start, end = rng.split("-")
                    ts = datetime.now().isoformat(timespec="seconds")
                    log.writerow([ts, t.id, ply, start, end])
                    csv_file.flush()
                    print(f"[{ts}] roll #{t.id}  ply={ply}  start={start}  end={end}")

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_fps, 1e-6)
        t_fps = now
        cv2.putText(frame, f"{fps:.1f} fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (fw, fh))
        writer.write(frame)
        if not args.no_show:
            cv2.imshow("RTracker OCR", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    csv_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
