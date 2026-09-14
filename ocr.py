"""PaddleOCR (PP-OCRv4) text recognizer on CPU via ONNX Runtime, restricted to the characters each field can contain."""
import math
import re

import cv2
import numpy as np
import onnxruntime as ort

# ply: number 0-99999, optionally with a UW/DW/UB/DB tag before or after it (e.g. "121 DW"); only the number is kept
PLY_RE = re.compile(r"^(?:[UD][WB])?(\d{1,5})(?:[UD][WB])?$")
RANGE_RE = re.compile(r"^\d{1,5}(\.\d{1,3})?-\d{1,5}(\.\d{1,3})?$")
CHARSET = {PLY_RE: "0123456789UWDB", RANGE_RE: "0123456789.-"}


def crop_rect(frame, rect, pad=4):
    """Cut out a rotated text box straightened so the text line runs left-to-right (may still be upside down).
    Crop x-axis = text direction u, crop y-axis (down) = v, perpendicular to it."""
    cx, cy, w, h, t = rect
    W, H = max(8, int(w + 2 * pad)), max(8, int(h + 2 * pad))
    ux, uy, vx, vy = math.cos(t), math.sin(t), -math.sin(t), math.cos(t)
    M = np.float32([[ux, vx, cx - W / 2 * ux - H / 2 * vx], [uy, vy, cy - W / 2 * uy - H / 2 * vy]])
    return cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def align_rect(rect, theta):
    """Point a text box the same way as theta (flip by 180 deg if needed). Near-vertical ply and range boxes can
    otherwise come out of the detector pointing in opposite directions."""
    cx, cy, w, h, t = rect
    return (cx, cy, w, h, t + math.pi) if math.cos(t - theta) < 0 else rect


def text_rotation(ply_rect, range_rect):
    """0 or 180: extra rotation that makes straightened crops upright. The ply number is written above the
    start-end, so if the ply lies on the crop's 'down' side (+v of the range box) the text is upside down."""
    t = range_rect[4]
    down = (ply_rect[0] - range_rect[0]) * -math.sin(t) + (ply_rect[1] - range_rect[1]) * math.cos(t)
    return 0 if down < 0 else 180


class TextReader:
    def __init__(self, model_path="models/ppocr_rec.onnx", threads=2):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        # Model output index 0 = CTC blank, 1..N = dictionary chars (stored in model metadata)
        chars = self.sess.get_modelmeta().custom_metadata_map["character"].splitlines()
        self.sets = {}  # charset -> (model output indices, characters)
        for allowed in CHARSET.values():
            idx = [0] + [i + 1 for i, c in enumerate(chars) if c in allowed]
            self.sets[allowed] = (idx, [""] + [chars[i - 1] for i in idx[1:]])

    def _read(self, crop, allowed):
        idx, chars = self.sets[allowed]
        h, w = crop.shape[:2]
        new_w = min(320, max(16, int(48 * w / h)))
        img = cv2.resize(crop, (new_w, 48)).astype(np.float32)
        img = (img[:, :, ::-1] / 255.0 - 0.5) / 0.5  # BGR->RGB, normalize to [-1, 1]
        x = np.zeros((1, 3, 48, 320), np.float32)
        x[0, :, :, :new_w] = img.transpose(2, 0, 1)
        probs = self.sess.run(None, {self.inp: x})[0][0][:, idx]  # keep only allowed chars
        best, conf = probs.argmax(1), probs.max(1)
        text, scores, prev = "", [], 0
        for k, p in zip(best, conf):
            if k != prev and k != 0:
                text += chars[k]
                scores.append(p)
            prev = k
        return text, float(np.mean(scores)) if scores else 0.0

    def read(self, crop, pattern, rotation=None):
        """Read a straightened text crop (see crop_rect). rotation (0/180) comes from the ply/range layout when
        known; otherwise both ways up are tried and the best valid reading wins. Returns (text, score) or (None, 0)."""
        if crop.size == 0:
            return None, 0.0
        if rotation is not None:
            tries = [crop if rotation == 0 else cv2.rotate(crop, cv2.ROTATE_180)]
        else:
            tries = [crop, cv2.rotate(crop, cv2.ROTATE_180)]
        best = (None, 0.0)
        for img in tries:
            text, score = self._read(img, CHARSET[pattern])
            if "-" not in text and text.count(".") == 3:  # short dash written like a dot: 31.3.34.1 -> 31.3-34.1
                a, b, c, d = text.split(".")
                text = f"{a}.{b}-{c}.{d}"
            m = pattern.match(text)
            if m and score > best[1]:
                best = (m.group(1) if pattern is PLY_RE else text, score)
        return best
