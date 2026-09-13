"""YOLOv8n detector. Hailo-8L (.hef) on the Raspberry Pi 5, Ultralytics (.pt) for testing on a PC.
Returns a list of (class_id, score, x1, y1, x2, y2) in frame pixels. Classes: 0=roll, 1=ply, 2=range, 3=any text."""
import cv2
import numpy as np

CLASSES = ["roll", "ply", "range"]


def letterbox(frame, w, h):
    s = min(w / frame.shape[1], h / frame.shape[0])
    nw, nh = int(frame.shape[1] * s), int(frame.shape[0] * s)
    px, py = (w - nw) // 2, (h - nh) // 2
    img = np.full((h, w, 3), 114, np.uint8)
    img[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh))
    return img, s, px, py


class HailoDetector:
    def __init__(self, hef_path, conf=0.4):
        from hailo_platform import VDevice, FormatType  # HailoRT 4.x (apt: hailo-all)
        self.conf = conf
        self.device = VDevice()
        self.model = self.device.create_infer_model(hef_path)
        self.model.set_batch_size(1)
        self.model.output().set_format_type(FormatType.FLOAT32)
        self.configured = self.model.configure()
        self.bindings = self.configured.create_bindings()
        self.h, self.w = self.model.input().shape[:2]

    def detect(self, frame):
        img, s, px, py = letterbox(frame, self.w, self.h)
        self.bindings.input().set_buffer(np.ascontiguousarray(img[:, :, ::-1]))  # BGR->RGB
        self.bindings.output().set_buffer(np.empty(self.model.output().shape, np.float32))
        self.configured.run([self.bindings], 1000)
        dets = []
        # NMS output: one array per class, rows = [ymin, xmin, ymax, xmax, score] normalized to model input
        for cls, rows in enumerate(self.bindings.output().get_buffer()):
            for y1, x1, y2, x2, score in rows:
                if score >= self.conf:
                    dets.append((cls, float(score),
                                 (x1 * self.w - px) / s, (y1 * self.h - py) / s,
                                 (x2 * self.w - px) / s, (y2 * self.h - py) / s))
        return dets


class UltralyticsDetector:
    def __init__(self, pt_path, conf=0.4):
        from ultralytics import YOLO
        self.model, self.conf = YOLO(pt_path), conf

    def detect(self, frame):
        b = self.model.predict(frame, conf=self.conf, imgsz=640, verbose=False)[0].boxes
        return [(int(c), float(s), *map(float, xyxy))
                for c, s, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]


class TextOnlyDetector:
    """No training needed (PC trial): PaddleOCR text detector finds text lines (class 3, read as ply or range
    by OCR); nearby lines are grouped into one box that stands in for the roll (class 0)."""
    text_only = True

    def __init__(self, onnx_path, conf=0.4, max_side=800):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.inp, self.conf, self.max_side = self.sess.get_inputs()[0].name, conf, max_side

    def detect(self, frame):
        fh, fw = frame.shape[:2]
        s = min(1.0, self.max_side / max(fh, fw))
        h, w = max(32, round(fh * s / 32) * 32), max(32, round(fw * s / 32) * 32)
        img = (cv2.resize(frame, (w, h)).astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        prob = self.sess.run(None, {self.inp: img.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
        contours, _ = cv2.findContours((prob > 0.3).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        lines = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh < 16 or prob[y:y + bh, x:x + bw].mean() < 0.3:
                continue
            d = 1.5 * bw * bh / (2 * (bw + bh))  # expand box like PaddleOCR's unclip
            lines.append([(x - d) * fw / w, (y - d) * fh / h, (x + bw + d) * fw / w, (y + bh + d) * fh / h])

        # group lines that are close together (ply number above start-end) into one box
        groups = []
        for box in lines:
            m = 0.7 * (box[3] - box[1])
            for g in groups:
                if box[0] - m < g[2] and box[2] + m > g[0] and box[1] - m < g[3] and box[3] + m > g[1]:
                    g[:] = [min(g[0], box[0]), min(g[1], box[1]), max(g[2], box[2]), max(g[3], box[3])]
                    break
            else:
                groups.append(list(box))
        return ([(0, 1.0, g[0] - 20, g[1] - 20, g[2] + 20, g[3] + 20) for g in groups] +
                [(3, 1.0, *b) for b in lines])


def load_detector(path, conf=0.4):
    if path.endswith(".hef"):
        return HailoDetector(path, conf)
    if path.endswith("ppocr_det.onnx"):
        return TextOnlyDetector(path, conf)
    return UltralyticsDetector(path, conf)
