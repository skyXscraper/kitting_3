"""YOLOv8n-OBB detector. Hailo-8L (.hef) on the Raspberry Pi 5, Ultralytics (.pt) for testing on a PC.
Returns a list of (class_id, score, rect) with rect = (cx, cy, w, h, theta) in frame pixels / radians,
normalized so w >= h (w runs along the text line). Classes (as labeled in Roboflow): 0=ply, 1=range, 2=roll;
3=any text (no-training mode)."""
import math

import cv2
import numpy as np

PLY, RANGE, ROLL, TEXT = 0, 1, 2, 3
CLASSES = ["ply", "range", "roll"]


def norm_rect(cx, cy, w, h, theta):
    """Long side along theta, theta in [-pi/2, pi/2) so the long axis always points right."""
    if h > w:
        w, h, theta = h, w, theta + math.pi / 2
    return float(cx), float(cy), float(w), float(h), (theta + math.pi / 2) % math.pi - math.pi / 2


def rect_points(rect):
    cx, cy, w, h, t = rect
    return cv2.boxPoints(((cx, cy), (w, h), math.degrees(t)))


def rect_bounds(rect):
    p = rect_points(rect)
    return (*p.min(0), *p.max(0))


def letterbox(frame, w, h):
    s = min(w / frame.shape[1], h / frame.shape[0])
    nw, nh = int(frame.shape[1] * s), int(frame.shape[0] * s)
    px, py = (w - nw) // 2, (h - nh) // 2
    img = np.full((h, w, 3), 114, np.uint8)
    img[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh))
    return img, s, px, py


def rotated_nms(dets, iou=0.5):
    """Class-wise rotated NMS on (cls, score, rect) tuples."""
    keep = []
    for c in {d[0] for d in dets}:
        ds = [d for d in dets if d[0] == c]
        boxes = [((r[0], r[1]), (r[2], r[3]), math.degrees(r[4])) for _, _, r in ds]
        idx = cv2.dnn.NMSBoxesRotated(boxes, [d[1] for d in ds], 0.0, iou)
        keep += [ds[i] for i in np.array(idx).flatten()]
    return keep


class HailoDetector:
    """HEF compiled by compile_hailo.py: 9 raw head outputs (box DFL 64ch, class 3ch, angle 1ch at 3 scales).
    Box/angle decoding and rotated NMS run here on the Pi CPU (only for anchors above the confidence threshold)."""

    def __init__(self, hef_path, conf=0.4):
        from hailo_platform import VDevice, FormatType  # HailoRT 4.x (apt: hailo-all)
        self.conf = conf
        self.device = VDevice()
        self.model = self.device.create_infer_model(hef_path)
        self.model.set_batch_size(1)
        for out in self.model.outputs:
            out.set_format_type(FormatType.FLOAT32)  # dequantized outputs
        self.configured = self.model.configure()
        self.bindings = self.configured.create_bindings()
        self.h, self.w = self.model.input().shape[:2]
        self.outputs = {o.name: o.shape for o in self.model.outputs}  # (H, W, C)

    def detect(self, frame):
        img, s, px, py = letterbox(frame, self.w, self.h)
        self.bindings.input().set_buffer(np.ascontiguousarray(img[:, :, ::-1]))  # BGR->RGB
        for name, shape in self.outputs.items():
            self.bindings.output(name).set_buffer(np.empty(shape, np.float32))
        self.configured.run([self.bindings], 1000)
        grids = {}  # grid size -> {"box"|"cls"|"ang": array}
        for name, shape in self.outputs.items():
            kind = {64: "box", 1: "ang"}.get(shape[2], "cls")
            grids.setdefault(shape[0], {})[kind] = self.bindings.output(name).get_buffer()
        dets = decode_obb(grids, self.h, self.conf)
        return [(c, sc, ((r[0] - px) / s, (r[1] - py) / s, r[2] / s, r[3] / s, r[4])) for c, sc, r in dets]


def decode_obb(grids, input_size, conf):
    """Decode YOLOv8-OBB raw head outputs (NHWC per grid) the same way Ultralytics does, then rotated NMS."""
    dets = []
    for g, out in grids.items():
        stride = input_size / g
        cls = 1 / (1 + np.exp(-out["cls"].reshape(g * g, -1)))
        score, c = cls.max(1), cls.argmax(1)
        keep = np.where(score >= conf)[0]
        if not len(keep):
            continue
        box = out["box"].reshape(g * g, 4, 16)[keep]
        box = np.exp(box - box.max(2, keepdims=True))
        dist = (box / box.sum(2, keepdims=True) * np.arange(16)).sum(2)  # DFL -> l, t, r, b
        ang = ((1 / (1 + np.exp(-out["ang"].reshape(g * g)[keep]))) - 0.25) * math.pi
        ax, ay = keep % g + 0.5, keep // g + 0.5
        xf, yf = (dist[:, 2] - dist[:, 0]) / 2, (dist[:, 3] - dist[:, 1]) / 2
        cx = (ax + xf * np.cos(ang) - yf * np.sin(ang)) * stride
        cy = (ay + xf * np.sin(ang) + yf * np.cos(ang)) * stride
        w, h = (dist[:, 0] + dist[:, 2]) * stride, (dist[:, 1] + dist[:, 3]) * stride
        dets += [(int(c[k]), float(score[k]), norm_rect(cx[i], cy[i], w[i], h[i], ang[i])) for i, k in enumerate(keep)]
    return rotated_nms(dets)


class UltralyticsDetector:
    def __init__(self, pt_path, conf=0.4):
        from ultralytics import YOLO
        self.model, self.conf = YOLO(pt_path), conf

    def detect(self, frame):
        obb = self.model.predict(frame, conf=self.conf, imgsz=640, verbose=False)[0].obb
        return [(int(c), float(s), norm_rect(*r))
                for c, s, r in zip(obb.cls.tolist(), obb.conf.tolist(), obb.xywhr.tolist())]


class TextOnlyDetector:
    """No training needed (PC trial): PaddleOCR text detector finds text lines (class 3, read as ply or range
    by OCR); nearby lines are grouped into one box that stands in for the roll (class 2)."""
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
        groups = []  # [box, member lines]
        for box in lines:
            m = 0.7 * min(box[2] - box[0], box[3] - box[1])
            for g, members in groups:
                if box[0] - m < g[2] and box[2] + m > g[0] and box[1] - m < g[3] and box[3] + m > g[1]:
                    g[:] = [min(g[0], box[0]), min(g[1], box[1]), max(g[2], box[2]), max(g[3], box[3])]
                    members.append(box)
                    break
            else:
                groups.append([list(box), [box]])

        def rect(b, pad=0):
            return norm_rect((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0] + 2 * pad, b[3] - b[1] + 2 * pad, 0.0)

        dets = []
        for g, members in groups:
            dets.append((ROLL, 1.0, rect(g, 20)))
            if len(members) >= 2:  # shortest line = ply number, longest line = start-end
                members.sort(key=lambda b: max(b[2] - b[0], b[3] - b[1]))
                dets += [(PLY, 1.0, rect(members[0])), (RANGE, 1.0, rect(members[-1]))]
                dets += [(TEXT, 1.0, rect(b)) for b in members[1:-1]]
            else:
                dets.append((TEXT, 1.0, rect(members[0])))
        return dets


def load_detector(path, conf=0.4):
    if path.endswith(".hef"):
        return HailoDetector(path, conf)
    if path.endswith("ppocr_det.onnx"):
        return TextOnlyDetector(path, conf)
    return UltralyticsDetector(path, conf)
