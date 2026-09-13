"""PaddleOCR (PP-OCRv4) text recognizer on CPU via ONNX Runtime, restricted to digits, '.' and '-'."""
import re
import cv2
import numpy as np
import onnxruntime as ort

PLY_RE = re.compile(r"^\d{1,3}$")
RANGE_RE = re.compile(r"^\d{1,4}(\.\d{1,3})?-\d{1,4}(\.\d{1,3})?$")
ALLOWED = "0123456789.-"


class TextReader:
    def __init__(self, model_path="models/ppocr_rec.onnx", threads=2):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        # Model output index 0 = CTC blank, 1..N = dictionary chars (stored in model metadata)
        chars = self.sess.get_modelmeta().custom_metadata_map["character"].splitlines()
        self.idx = [0] + [i + 1 for i, c in enumerate(chars) if c in ALLOWED]
        self.chars = [""] + [chars[i - 1] for i in self.idx[1:]]

    def _read(self, crop):
        h, w = crop.shape[:2]
        new_w = min(320, max(16, int(48 * w / h)))
        img = cv2.resize(crop, (new_w, 48)).astype(np.float32)
        img = (img[:, :, ::-1] / 255.0 - 0.5) / 0.5  # BGR->RGB, normalize to [-1, 1]
        x = np.zeros((1, 3, 48, 320), np.float32)
        x[0, :, :, :new_w] = img.transpose(2, 0, 1)
        probs = self.sess.run(None, {self.inp: x})[0][0][:, self.idx]  # keep only allowed chars
        best, conf = probs.argmax(1), probs.max(1)
        text, scores, prev = "", [], 0
        for k, p in zip(best, conf):
            if k != prev and k != 0:
                text += self.chars[k]
                scores.append(p)
            prev = k
        return text, float(np.mean(scores)) if scores else 0.0

    def read(self, crop, pattern):
        """Read a text crop; tries rotations for vertical/upside-down text. Returns (text, score) or (None, 0)."""
        if crop.size == 0:
            return None, 0.0
        if crop.shape[0] > crop.shape[1] * 1.2:  # vertical text on a standing roll
            tries = [cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)]
        else:
            tries = [crop, cv2.rotate(crop, cv2.ROTATE_180)]
        best = (None, 0.0)
        for img in tries:
            text, score = self._read(img)
            if "-" not in text and text.count(".") == 3:  # short dash written like a dot: 31.3.34.1 -> 31.3-34.1
                a, b, c, d = text.split(".")
                text = f"{a}.{b}-{c}.{d}"
            if pattern.match(text) and score > best[1]:
                best = (text, score)
        return best
