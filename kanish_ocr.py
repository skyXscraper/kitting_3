from __future__ import annotations

import argparse
import csv
import json
import queue
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Text / master helpers (self-contained — does not modify ocr_single_roll.py)
# ---------------------------------------------------------------------------


def ascii_label(text: str, max_len: int = 40) -> str:
    """OpenCV Hershey fonts only draw ASCII — strip the rest (avoids ??? overlays)."""
    t = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in str(text or ""))
    t = t.replace("?", "").strip()  # drop undecodable junk entirely
    if not t:
        return ""
    return t[:max_len]


def looks_like_plant_label(text: str) -> bool:
    """Ignore fixed machine/plate text (e.g. MCC-17) — not ply handwriting."""
    t = (text or "").strip().upper().replace(" ", "")
    if re.fullmatch(r"MCC[-_]?\d+", t):
        return True
    if t in {"MCC", "M/C", "MC"}:
        return True
    return False


def normalize_ocr_token(text: str) -> str:
    """Map common OCR junk toward digit/range grammar."""
    t = (text or "").strip()
    t = t.replace(":", ".").replace("≠", "-").replace("=", "-")
    t = t.replace("O", "0").replace("o", "0").replace("Q", "9").replace("q", "9")
    t = t.replace("—", "-").replace("–", "-").replace("_", "-")
    # Collapse "3.1.3-37.1" style → keep as soup digits for recovery
    return t


def sanitize(text: str) -> str:
    t = normalize_ocr_token(text)
    t = re.sub(r"[^\d.\- ]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def load_master_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def master_index(master: list[dict]) -> dict[str, dict]:
    by_ply = {}
    for r in master:
        key = str(r.get("ply_no") or r.get("serial") or "").strip()
        if key:
            by_ply[key] = r
    return by_ply


def digit_confusion_candidates(text: str) -> list[str]:
    """OCR 4→try 9; OCR 9→try 7 (same rule as EasyOCR path)."""
    text = sanitize(text)
    if not text:
        return []
    idxs = [i for i, ch in enumerate(text) if ch in ("4", "9")]
    out = {text}
    if not idxs:
        return [text]
    n = len(idxs)
    for mask in range(1, 1 << n):
        chars = list(text)
        for bit in range(n):
            if mask & (1 << bit):
                i = idxs[bit]
                if chars[i] == "4":
                    chars[i] = "9"
                elif chars[i] == "9":
                    chars[i] = "7"
        out.add("".join(chars))
    return sorted(out)


def normalize_range_str(s: str | None) -> str:
    return sanitize(s or "").replace(" ", "")


def ranges_equal(a: str | None, b: str | None) -> bool:
    return normalize_range_str(a) == normalize_range_str(b)


def parse_range_floats(s: str | None) -> tuple[float, float] | None:
    s = normalize_range_str(s)
    m = re.fullmatch(r"(\d+\.?\d*)-(\d+\.?\d*)", s)
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return None


def range_distance(ocr_range: str | None, expected: str | None) -> float:
    a = parse_range_floats(ocr_range)
    b = parse_range_floats(expected)
    if not a or not b:
        return 1e9
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def soup_to_range_candidates(text: str) -> list[str]:
    digits = re.sub(r"\D", "", sanitize(text))
    if len(digits) < 5:
        return []
    out = []
    for split in range(2, len(digits) - 1):
        left, right = digits[:split], digits[split:]
        for lf in (1, 2, 3):
            if lf >= len(left):
                continue
            for rf in (1, 2, 3):
                if rf >= len(right):
                    continue
                a = f"{left[:-lf]}.{left[-lf:]}"
                b = f"{right[:-rf]}.{right[-rf:]}"
                try:
                    fa, fb = float(a), float(b)
                except ValueError:
                    continue
                if 0 < fa < 200 and 0 < fb < 200 and fa < fb:
                    out.append(f"{a}-{b}")
    seen: set[str] = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def apply_confusion_master_check(
    ply_ocr: str | None,
    range_ocr: str | None,
    length_frags: list[str],
    master: list[dict],
    soup_ranges: list[str] | None = None,
) -> dict:
    by_ply = master_index(master)
    result = {
        "applied": False,
        "ply_corrected": None,
        "range_corrected": None,
        "rule": "OCR 4→try 9; OCR 9→try 7; accept only if in master_list",
        "candidates_tried": [],
        "note": None,
        "match": None,
        "range_ok": False,
        "expected_range": None,
    }
    if not ply_ocr or not by_ply:
        return result

    ply_options = digit_confusion_candidates(ply_ocr)
    range_options: set[str] = set()
    if range_ocr:
        range_options.add(normalize_range_str(range_ocr))
        for c in digit_confusion_candidates(range_ocr):
            range_options.add(normalize_range_str(c))

    frags = [sanitize(t) for t in length_frags if re.fullmatch(r"\d+\.\d+", sanitize(t))]
    frags = list(dict.fromkeys(frags))
    if len(frags) >= 2:
        for a in digit_confusion_candidates(frags[0]):
            for b in digit_confusion_candidates(frags[1]):
                try:
                    fa, fb = float(a), float(b)
                    lo, hi = (a, b) if fa <= fb else (b, a)
                    range_options.add(f"{lo}-{hi}")
                except ValueError:
                    range_options.add(f"{a}-{b}")

    for s in soup_ranges or []:
        range_options.add(normalize_range_str(s))
        for c in digit_confusion_candidates(s):
            range_options.add(normalize_range_str(c))

    hits = []
    for ply in ply_options:
        result["candidates_tried"].append(ply)
        row = by_ply.get(ply)
        if not row:
            continue
        expected = normalize_range_str(f"{row.get('start')}-{row.get('end')}")
        range_ok = any(ranges_equal(ro, expected) for ro in range_options)
        hits.append(
            {
                "ply": ply,
                "row": row,
                "expected_range": expected,
                "range_ok": range_ok,
            }
        )

    if not hits:
        range_hits = []
        for ro in range_options:
            for key, row in by_ply.items():
                expected = normalize_range_str(f"{row.get('start')}-{row.get('end')}")
                if ranges_equal(ro, expected):
                    range_hits.append((key, row, expected))
        unique = {h[0] for h in range_hits}
        if len(unique) == 1:
            key, row, expected = range_hits[0]
            return {
                "applied": True,
                "ply_corrected": key,
                "range_corrected": expected,
                "match": row,
                "range_ok": True,
                "expected_range": expected,
                "candidates_tried": list(ply_options),
                "rule": result["rule"],
                "note": (
                    f"Ply OCR '{ply_ocr}' not in master after remap; "
                    f"unique start-end match → ply '{key}' ({expected})"
                ),
            }
        return result

    def best_dist(h: dict) -> float:
        d = range_distance(range_ocr, h["expected_range"])
        for ro in range_options:
            d = min(d, range_distance(ro, h["expected_range"]))
        return d

    hits.sort(key=lambda h: (not h["range_ok"], best_dist(h)))
    best = hits[0]
    soft = False
    if not best["range_ok"] and len(hits) > 1:
        dists = sorted((best_dist(h), h) for h in hits)
        if dists[0][0] >= 1e8 or (len(dists) > 1 and dists[1][0] - dists[0][0] < 2.0):
            result["note"] = (
                f"Ambiguous remaps {[h['ply'] for h in hits]} without start-end match; "
                f"keeping OCR ply '{ply_ocr}'"
            )
            return result
        best = dists[0][1]
        if dists[0][0] <= 5.0:
            soft = True
    elif not best["range_ok"] and best_dist(best) <= 5.0:
        soft = True

    range_ok = best["range_ok"] or soft
    applied = best["ply"] != ply_ocr or range_ok
    result.update(
        {
            "applied": applied,
            "ply_corrected": best["ply"],
            "range_corrected": best["expected_range"] if range_ok else None,
            "match": best["row"],
            "range_ok": range_ok,
            "expected_range": best["expected_range"],
            "note": (
                f"OCR ply '{ply_ocr}' → '{best['ply']}' via 4→9 / 9→7; "
                + (
                    "start-end matched master"
                    if best["range_ok"]
                    else (
                        "start-end nearest master row (soft match)"
                        if soft
                        else "ply in master but start-end not matched"
                    )
                )
            ),
        }
    )
    return result


# ---------------------------------------------------------------------------
# Pattern assembly from RapidOCR / Paddle boxes
# ---------------------------------------------------------------------------


def box_from_points(box) -> tuple[int, int, int, int]:
    """4-point polygon → axis-aligned xyxy."""
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def parse_roi_arg(roi: str | None, w: int, h: int) -> tuple[int, int, int, int] | None:
    """
    ROI as:
      - preset: bottom_right | lower_half | right_half | full
      - fractions: x1,y1,x2,y2 in 0..1  e.g. 0.5,0.4,1,1
      - pixels: 640,300,1280,720
    """
    if not roi or roi.lower() in ("full", "none", "-"):
        return None
    key = roi.strip().lower().replace("-", "_")
    presets = {
        "bottom_right": (0.48, 0.38, 1.0, 1.0),
        "lower_right": (0.48, 0.38, 1.0, 1.0),
        "lower_half": (0.0, 0.45, 1.0, 1.0),
        "right_half": (0.45, 0.0, 1.0, 1.0),
        "write_desk": (0.40, 0.35, 0.98, 0.95),
    }
    if key in presets:
        x1f, y1f, x2f, y2f = presets[key]
        return (
            int(w * x1f),
            int(h * y1f),
            int(w * x2f),
            int(h * y2f),
        )
    parts = [p.strip() for p in roi.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError(f"Bad --roi '{roi}'. Use preset or x1,y1,x2,y2")
    vals = [float(p) for p in parts]
    if all(0.0 <= v <= 1.5 for v in vals) and max(vals) <= 1.5:
        x1, y1, x2, y2 = (
            int(w * vals[0]),
            int(h * vals[1]),
            int(w * vals[2]),
            int(h * vals[3]),
        )
    else:
        x1, y1, x2, y2 = [int(v) for v in vals]
    x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
    if x2 - x1 < 20 or y2 - y1 < 20:
        raise ValueError(f"ROI too small: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def crop_and_zoom(
    bgr: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    zoom: float,
) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    """
    Crop ROI then upscale for OCR.
    Returns (work_image, roi_xyxy_on_full_frame, scale_work_to_full).
    Boxes from OCR on work_image map back: full = work / scale + roi_origin
    """
    h, w = bgr.shape[:2]
    if roi is None:
        roi = (0, 0, w, h)
    x1, y1, x2, y2 = roi
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return bgr, (0, 0, w, h), 1.0
    z = max(1.0, float(zoom))
    if z != 1.0:
        work = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    else:
        work = crop
    # scale: work pixel → full-frame pixel offset from (x1,y1)
    return work, (x1, y1, x2, y2), z


def map_box_to_full(
    box: tuple[int, int, int, int],
    roi: tuple[int, int, int, int],
    zoom: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    ox, oy = roi[0], roi[1]
    z = max(zoom, 1e-6)
    return (
        int(x1 / z + ox),
        int(y1 / z + oy),
        int(x2 / z + ox),
        int(y2 / z + oy),
    )


def check_two_row_layout(dets: list[dict]) -> dict:
    """
    Explicit handwriting check:
      Top row  → ply number (integer)
      Bottom   → start-end (range or two length fragments)
    Uses vertical order of boxes (smaller y = higher on image).
    """
    useful = []
    for d in dets:
        raw = str(d["text"])
        if looks_like_plant_label(raw):
            continue
        text = sanitize(raw)
        if not text and not re.search(r"\d", normalize_ocr_token(raw)):
            continue
        useful.append(
            {
                "text": text or sanitize(normalize_ocr_token(raw)),
                "raw": raw,
                "conf": float(d["conf"]),
                "box": d["box"],
                "yc": (d["box"][1] + d["box"][3]) / 2.0,
            }
        )
    if not useful:
        return {
            "ok": False,
            "top_row": None,
            "bottom_row": None,
            "note": "No digit-like text in ROI (need 2 rows: ply then start-end)",
        }

    useful.sort(key=lambda x: (x["yc"], x["box"][0]))
    # Cluster into top / bottom by y gap if >=2 boxes
    top_row = None
    bottom_row = None
    if len(useful) == 1:
        only = useful[0]["text"]
        if re.fullmatch(r"\d{1,4}", only):
            top_row = only
            note = "Only 1 row seen (ply-like); missing start-end row"
        else:
            bottom_row = only
            note = "Only 1 row seen (range-like); missing ply row"
        return {"ok": False, "top_row": top_row, "bottom_row": bottom_row, "note": note}

    # Split at largest vertical gap
    gaps = [(useful[i + 1]["yc"] - useful[i]["yc"], i) for i in range(len(useful) - 1)]
    gaps.sort(reverse=True)
    split_i = gaps[0][1]
    top_group = useful[: split_i + 1]
    bot_group = useful[split_i + 1 :]
    # If gap tiny, use first as top and rest as bottom
    if gaps[0][0] < 8 and len(useful) >= 2:
        top_group = [useful[0]]
        bot_group = useful[1:]

    def group_text(group: list[dict]) -> str:
        parts = [g["text"] for g in sorted(group, key=lambda x: x["box"][0]) if g["text"]]
        return " ".join(parts).strip()

    top_txt = group_text(top_group)
    bot_txt = group_text(bot_group)

    top_ok = bool(re.fullmatch(r"\d{1,4}", sanitize(top_txt).replace(" ", "")))
    bot_san = sanitize(bot_txt).replace(" ", "")
    bot_ok = bool(
        re.fullmatch(r"\d+\.\d+-\d+\.\d+", bot_san)
        or re.fullmatch(r"\d+-\d+\.\d+", bot_san)
        or re.fullmatch(r"\d+\.\d+-\d+", bot_san)
        or re.fullmatch(r"\d+\.\d+\s+\d+\.\d+", sanitize(bot_txt))
        or (
            len([g for g in bot_group if re.fullmatch(r"\d+\.\d+", g["text"])]) >= 2
        )
    )

    # Recover bottom range from two decimals in bottom group
    bottom_range = None
    if re.fullmatch(r"\d+\.\d+-\d+\.\d+", bot_san):
        bottom_range = bot_san
        bot_ok = True
    else:
        decs = [g["text"] for g in bot_group if re.fullmatch(r"\d+\.\d+", g["text"])]
        if len(decs) >= 2:
            try:
                a, b = decs[0], decs[1]
                lo, hi = (a, b) if float(a) <= float(b) else (b, a)
                bottom_range = f"{lo}-{hi}"
                bot_ok = True
            except ValueError:
                pass
        if bottom_range is None and bot_san:
            m = re.search(r"(\d+\.\d+)-(\d+\.\d+)", bot_san)
            if m:
                bottom_range = f"{m.group(1)}-{m.group(2)}"
                bot_ok = True
            else:
                m2 = re.search(r"(\d+\.?\d*)-(\d+\.?\d*)", bot_san)
                if m2:
                    bottom_range = f"{m2.group(1)}-{m2.group(2)}"
                    bot_ok = True
        # Digit soup on bottom row e.g. 79204 / 7920.4 → try start-end recoveries
        if bottom_range is None:
            for g in bot_group:
                for s in soup_to_range_candidates(g["raw"] if g.get("raw") else g["text"]):
                    if re.fullmatch(r"\d+\.\d+-\d+\.\d+", s):
                        bottom_range = s
                        bot_ok = True
                        break
                if bottom_range:
                    break

    ok = top_ok and bot_ok
    note = (
        "2-row OK: top=ply, bottom=start-end"
        if ok
        else (
            f"2-row check failed: top='{top_txt}' (ply_int={top_ok}), "
            f"bottom='{bot_txt}' (range={bot_ok})"
        )
    )
    return {
        "ok": ok,
        "top_row": sanitize(top_txt) if top_ok else top_txt,
        "bottom_row": bottom_range or bot_txt,
        "top_ok": top_ok,
        "bottom_ok": bot_ok,
        "note": note,
    }


def assemble_detections(dets: list[dict]) -> dict:
    """
    dets: {text, conf, box=(x1,y1,x2,y2)}
    Pattern: top integer = ply; decimals / ranges = start-end.
    Also reports explicit two_row_layout check.
    """
    two_row = check_two_row_layout(dets)

    ply_cands = []
    length_frags = []
    range_cands = []
    soup_ranges: list[str] = []

    for d in dets:
        raw = str(d["text"])
        if looks_like_plant_label(raw):
            continue
        text = sanitize(raw)
        if not text:
            norm = re.sub(r"[^\d.\-]", "", normalize_ocr_token(raw))
            if len(re.sub(r"\D", "", norm)) >= 5:
                soup_ranges.extend(soup_to_range_candidates(norm))
            continue
        box = d["box"]
        conf = float(d["conf"])
        if re.fullmatch(r"\d{1,4}", text):
            ply_cands.append({"text": text, "conf": conf, "box": box})
        elif re.fullmatch(r"\d+\.\d+-\d+\.\d+", text.replace(" ", "")) or re.fullmatch(
            r"\d+-\d+", text.replace(" ", "")
        ):
            range_cands.append({"text": text.replace(" ", ""), "conf": conf, "box": box})
        elif re.fullmatch(r"\d+\.\d+", text):
            length_frags.append({"text": text, "conf": conf, "box": box})
        else:
            digits = re.sub(r"\D", "", text)
            if len(digits) >= 5:
                soup_ranges.extend(soup_to_range_candidates(text))
            if text.count(".") >= 2 or re.search(r"\d+\.\d+\.\d+", normalize_ocr_token(raw)):
                soup_ranges.extend(soup_to_range_candidates(normalize_ocr_token(raw)))

    # Prefer two-row geometry when available
    ply_no = None
    ply_conf = 0.0
    range_str = None
    if two_row.get("top_ok") and two_row.get("top_row"):
        ply_no = sanitize(str(two_row["top_row"]))
        ply_conf = 0.9
    else:
        ply_cands.sort(key=lambda x: (0 if len(x["text"]) >= 2 else 1, x["box"][1], -x["conf"]))
        ply_no = ply_cands[0]["text"] if ply_cands else None
        ply_conf = ply_cands[0]["conf"] if ply_cands else 0.0

    if two_row.get("bottom_ok") and two_row.get("bottom_row"):
        br = sanitize(str(two_row["bottom_row"])).replace(" ", "")
        if re.search(r"\d", br):
            range_str = br if "-" in br else None
            if range_str is None:
                range_str = two_row["bottom_row"]

    if range_str is None and len(length_frags) >= 2:
        length_frags.sort(key=lambda x: (x["box"][0], -x["conf"]))
        texts = []
        for f in length_frags:
            if f["text"] not in texts:
                texts.append(f["text"])
        if len(texts) >= 2:
            a, b = texts[0], texts[1]
            try:
                lo, hi = (a, b) if float(a) <= float(b) else (b, a)
                range_str = f"{lo}-{hi}"
            except ValueError:
                range_str = f"{a}-{b}"

    good_ranges = [
        r["text"]
        for r in sorted(range_cands, key=lambda x: -x["conf"])
        if re.fullmatch(r"\d+\.\d+-\d+\.\d+", r["text"])
    ]
    if range_str is None and good_ranges:
        range_str = good_ranges[0]
    if range_str is None and range_cands:
        range_str = range_cands[0]["text"]

    soup_ranges = list(dict.fromkeys(soup_ranges))
    good_soups = [s for s in soup_ranges if re.fullmatch(r"\d+\.\d+-\d+\.\d+", s)]
    if range_str is None and good_soups:
        range_str = good_soups[0]
    elif range_str and not re.fullmatch(r"\d+\.\d+-\d+\.\d+", str(range_str).replace(" ", "")) and good_soups:
        range_str = good_soups[0]

    return {
        "ply_no_ocr": ply_no,
        "ply_no_confidence": ply_conf,
        "start_end_ocr": range_str,
        "length_fragments": [f["text"] for f in length_frags],
        "soup_range_candidates": soup_ranges[:30],
        "two_row_layout": two_row,
        "raw_detections": dets,
    }


# ---------------------------------------------------------------------------
# OCR engine — ONNX PP-OCR (Paddle) via RapidOCR
# ---------------------------------------------------------------------------


class PaddleOnnxEngine:
    """
    PP-OCR via RapidOCR ONNX (PaddleOCR-family models).
    Same backend on Mac and Pi → same results (CPU ONNX).
    """

    name = "rapidocr-onnx-ppocr"

    def __init__(self):
        try:
            from rapidocr import RapidOCR
        except ImportError as e:
            raise SystemExit(
                "Missing rapidocr. Install with:\n"
                "  pip install -r requirements-paddle-video.txt\n"
                f"Original error: {e}"
            ) from e
        # Bundled PP-OCR ONNX det+cls+rec (engine_name=onnxruntime)
        self.engine = RapidOCR()

    def run(self, bgr: np.ndarray) -> list[dict]:
        # RapidOCR 3.x → RapidOCROutput(boxes, txts, scores, ...)
        out = self.engine(bgr)
        dets: list[dict] = []

        # Legacy rapidocr-onnxruntime returned (list, elapse)
        if isinstance(out, tuple):
            result, _elapse = out
            if not result:
                return dets
            for item in result:
                if len(item) < 3:
                    continue
                dets.append(
                    {
                        "text": str(item[1]),
                        "conf": float(item[2]),
                        "box": box_from_points(item[0]),
                    }
                )
            return dets

        boxes = getattr(out, "boxes", None)
        txts = getattr(out, "txts", None) or ()
        scores = getattr(out, "scores", None) or ()
        if boxes is None:
            return dets
        for i, text in enumerate(txts):
            conf = float(scores[i]) if i < len(scores) else 0.0
            dets.append(
                {
                    "text": str(text),
                    "conf": conf,
                    "box": box_from_points(boxes[i]),
                }
            )
        return dets


def draw_overlay(
    bgr: np.ndarray,
    dets: list[dict],
    *,
    frame_idx: int,
    display_fps: float,
    ocr_fps: float | None,
    ply: str | None,
    rng: str | None,
    status: str | None,
    ocr_age_frames: int,
    roi: tuple[int, int, int, int] | None = None,
    two_row: dict | None = None,
    frozen: dict | None = None,
    vote_status: str | None = None,
) -> np.ndarray:
    """Live/annotated overlay: ROI + boxes + FPS + last OCR text."""
    out = bgr.copy()
    h, w = out.shape[:2]

    if roi is not None:
        rx1, ry1, rx2, ry2 = [int(v) for v in roi]
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2)
        cv2.putText(
            out,
            "OCR ROI",
            (rx1 + 6, max(20, ry1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )

    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        is_plant = looks_like_plant_label(str(d["text"]))
        color = (120, 120, 120) if is_plant else (0, 220, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = ascii_label(str(d["text"]))
        if not label:
            continue
        if d.get("conf") is not None:
            label = f"{label} ({float(d['conf']):.2f})"
        cv2.putText(
            out,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    banner_h = 128
    cv2.rectangle(out, (0, 0), (w, banner_h), (0, 0, 0), -1)
    ocr_fps_txt = f"{ocr_fps:.1f}" if ocr_fps and ocr_fps > 0 else "-"
    line1 = f"FPS display: {display_fps:.1f}   |   OCR: {ocr_fps_txt} fps   |   frame {frame_idx}"
    line2 = f"ply={ply or '-'}   range={rng or '-'}   status={status or '-'}"
    if ocr_age_frames > 0:
        line2 += f"   (OCR {ocr_age_frames}f ago)"
    tr = two_row or {}
    line3 = (
        f"2-row: {'OK' if tr.get('ok') else 'NO'}  "
        f"top={tr.get('top_row') or '-'}  bottom={tr.get('bottom_row') or '-'}"
    )
    if frozen:
        line4 = (
            f"PLY DETECTED: {frozen.get('ply')}   |   BIN: {frozen.get('bin') or '-'}   |   "
            f"range {frozen.get('range') or '-'}   [{frozen.get('reason')}]"
        )
        line4_color = (0, 255, 0)
    else:
        line4 = vote_status or "voting..."
        line4_color = (0, 200, 255)
    cv2.putText(out, line1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, line2, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        out,
        line3,
        (12, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 128) if tr.get("ok") else (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(out, line4, (12, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.62, line4_color, 2, cv2.LINE_AA)
    if frozen:
        # Large freeze badge
        badge = f"PLY {frozen.get('ply')}  ->  BIN {frozen.get('bin') or '-'}"
        cv2.rectangle(out, (w // 2 - 220, h - 70), (w // 2 + 220, h - 20), (0, 120, 0), -1)
        cv2.putText(
            out,
            badge,
            (w // 2 - 200, h - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    useful = [d for d in dets if not looks_like_plant_label(str(d["text"]))]
    y = banner_h + 24
    cv2.putText(out, "Detected:", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    y += 22
    if not useful:
        cv2.putText(
            out,
            "(none / waiting OCR)",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
    else:
        for d in useful[:8]:
            t = ascii_label(str(d["text"]))
            if not t:
                continue
            cv2.putText(
                out,
                f"- {t}",
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 128),
                1,
                cv2.LINE_AA,
            )
            y += 20

    cv2.putText(
        out,
        "q=quit  space=pause",
        (12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return out


def process_frame(
    engine: PaddleOnnxEngine,
    bgr: np.ndarray,
    master: list[dict],
    max_side: int,
    roi_xyxy: tuple[int, int, int, int] | None = None,
    zoom: float = 2.5,
) -> dict:
    h, w = bgr.shape[:2]
    if roi_xyxy is None:
        roi_xyxy = (0, 0, w, h)

    work, roi_used, z = crop_and_zoom(bgr, roi_xyxy, zoom)

    # Optional cap on OCR input size after zoom (Pi RAM / speed)
    wh, ww = work.shape[:2]
    extra = 1.0
    if max(wh, ww) > max_side * 2:
        extra = (max_side * 2) / float(max(wh, ww))
        work = cv2.resize(work, (int(ww * extra), int(wh * extra)), interpolation=cv2.INTER_AREA)
        z = z * extra

    t0 = time.perf_counter()
    dets = engine.run(work)
    dt = time.perf_counter() - t0

    for d in dets:
        d["box"] = map_box_to_full(tuple(d["box"]), roi_used, z)

    assembled = assemble_detections(dets)
    confusion = apply_confusion_master_check(
        assembled["ply_no_ocr"],
        assembled["start_end_ocr"],
        assembled["length_fragments"],
        master,
        assembled["soup_range_candidates"],
    )

    ply_ocr = assembled["ply_no_ocr"]
    range_ocr = assembled["start_end_ocr"]
    ply_final = confusion["ply_corrected"] if confusion.get("applied") else ply_ocr
    range_final = (
        confusion["range_corrected"]
        if confusion.get("applied") and confusion.get("range_corrected")
        else range_ocr
    )
    two_row = assembled.get("two_row_layout") or {}

    if confusion.get("applied") and confusion.get("match"):
        status = "corrected_via_4to9_9to7_and_master"
    elif ply_ocr and master_index(master).get(ply_ocr):
        status = "ply_found_exact_ocr"
    elif two_row.get("ok"):
        status = "two_row_ok_not_in_master"
    elif ply_ocr:
        status = "ply_not_in_master"
    else:
        status = "no_ply_detected"

    return {
        "engine": engine.name,
        "inference_s": round(dt, 3),
        "roi": list(roi_used),
        "zoom": zoom,
        "detections": [
            {"text": d["text"], "conf": d["conf"], "box": list(d["box"])} for d in dets
        ],
        "assembled": {
            "ply_no_ocr": ply_ocr,
            "start_end_ocr": range_ocr,
            "ply_no_confidence": assembled["ply_no_confidence"],
            "length_fragments": assembled["length_fragments"],
            "soup_range_candidates": assembled["soup_range_candidates"],
            "two_row_layout": two_row,
        },
        "final": {
            "ply_no_ocr": ply_ocr,
            "start_end_ocr": range_ocr,
            "ply_no_final": ply_final,
            "start_end_final": range_final,
            "master_status": status,
            "two_row_ok": bool(two_row.get("ok")),
            "two_row_note": two_row.get("note"),
            "confusion_note": confusion.get("note"),
            "operator_action": (
                "auto"
                if status in ("corrected_via_4to9_9to7_and_master", "ply_found_exact_ocr")
                else ("confirm" if ply_ocr or two_row.get("ok") else "manual")
            ),
        },
        "confusion": {
            "applied": confusion.get("applied"),
            "ply_corrected": confusion.get("ply_corrected"),
            "range_corrected": confusion.get("range_corrected"),
            "note": confusion.get("note"),
        },
    }



class PlyRangeConfirmer:
    """
    Because OCR often sees only ONE line per frame:
      1) See ply in master  -> PENDING (wait for start-end)
      2) Later see start-end that matches that ply's master range -> FROZEN
      3) Stay frozen until a *different* ply (in master) is detected

    Same-frame ply+matching-range also freezes immediately.
    """

    def __init__(self, by_ply: dict[str, dict], soft_range_tol: float = 3.0):
        self.by_ply = by_ply
        self.soft_range_tol = soft_range_tol
        self.pending: dict | None = None  # waiting for range confirm
        self.frozen: dict | None = None

    def _expected_range(self, ply: str) -> str | None:
        row = self.by_ply.get(str(ply))
        if not row:
            return None
        return normalize_range_str(f"{row.get('start')}-{row.get('end')}")

    def _range_matches_expected(self, range_ocr: str | None, expected: str | None) -> bool:
        if not range_ocr or not expected:
            return False
        for cand in digit_confusion_candidates(normalize_range_str(range_ocr)):
            if ranges_equal(cand, expected):
                return True
            if range_distance(cand, expected) <= self.soft_range_tol:
                return True
        return False

    def _ply_in_master(self, ply: str | None) -> dict | None:
        if not ply:
            return None
        return self.by_ply.get(str(ply))

    def _new_ply_signal(self, sample: dict) -> str | None:
        """A clear multi-digit ply reading that exists in master."""
        ply = sample.get("ply")
        if not ply or not re.fullmatch(r"\d{2,4}", str(ply)):
            return None
        if self._ply_in_master(ply):
            return str(ply)
        return None

    def observe(self, sample: dict) -> str | None:
        """
        Returns: 'frozen' | 'pending' | 'switched' | 'cleared:...' | None
        """
        ply = sample.get("ply")
        rng = sample.get("range")
        row = self._ply_in_master(ply) if ply else None

        # --- Already frozen: hold until a *new* master ply appears ---
        if self.frozen is not None:
            new_ply = self._new_ply_signal(sample)
            if new_ply and new_ply != str(self.frozen.get("ply")):
                old = self.frozen.get("ply")
                exp = self._expected_range(new_ply)
                # If this frame also has matching range, freeze new immediately
                if self._range_matches_expected(rng, exp):
                    nrow = self.by_ply[new_ply]
                    self.frozen = {
                        "ply": new_ply,
                        "range": exp,
                        "bin": nrow.get("no_of_ply"),
                        "row": nrow,
                        "reason": "ply_and_range_same_frame",
                        "frame_index": sample.get("frame_index"),
                    }
                    self.pending = None
                    return "frozen"
                self.pending = {
                    "ply": new_ply,
                    "expected_range": exp,
                    "bin": self.by_ply[new_ply].get("no_of_ply"),
                    "row": self.by_ply[new_ply],
                    "frame_index": sample.get("frame_index"),
                }
                self.frozen = None
                return f"switched:{old}->{new_ply}"
            return None

        # --- Pending: have ply, waiting for matching start-end ---
        if self.pending is not None:
            pend_ply = str(self.pending["ply"])
            expected = self.pending.get("expected_range")
            # Switch pending if a different master ply shows up
            new_ply = self._new_ply_signal(sample)
            if new_ply and new_ply != pend_ply:
                self.pending = {
                    "ply": new_ply,
                    "expected_range": self._expected_range(new_ply),
                    "bin": self.by_ply[new_ply].get("no_of_ply"),
                    "row": self.by_ply[new_ply],
                    "frame_index": sample.get("frame_index"),
                }
                return "pending"
            # Range-first pending: confirm when ply OCR matches
            if self.pending.get("from_range_first"):
                if ply and str(ply) == pend_ply:
                    self.frozen = {
                        "ply": pend_ply,
                        "range": expected,
                        "bin": self.pending.get("bin"),
                        "row": self.pending.get("row"),
                        "reason": "range_then_ply_confirmed",
                        "frame_index": sample.get("frame_index"),
                    }
                    self.pending = None
                    return "frozen"
                return None
            if self._range_matches_expected(rng, expected):
                self.frozen = {
                    "ply": pend_ply,
                    "range": expected,
                    "bin": self.pending.get("bin"),
                    "row": self.pending.get("row"),
                    "reason": "ply_then_range_confirmed",
                    "frame_index": sample.get("frame_index"),
                    "confirmed_range_ocr": rng,
                }
                self.pending = None
                return "frozen"
            # Same ply seen again — stay pending
            return None

        # --- Idle: look for ply in master (optionally with range same frame) ---
        if row and ply:
            expected = self._expected_range(str(ply))
            if self._range_matches_expected(rng, expected):
                self.frozen = {
                    "ply": str(ply),
                    "range": expected,
                    "bin": row.get("no_of_ply"),
                    "row": row,
                    "reason": "ply_and_range_same_frame",
                    "frame_index": sample.get("frame_index"),
                    "confirmed_range_ocr": rng,
                }
                return "frozen"
            self.pending = {
                "ply": str(ply),
                "expected_range": expected,
                "bin": row.get("no_of_ply"),
                "row": row,
                "frame_index": sample.get("frame_index"),
            }
            return "pending"

        # Range-only: if unique master start-end, start pending that ply (wait for ply OCR)
        if rng and not ply:
            hits = []
            for key, r in self.by_ply.items():
                exp = normalize_range_str(f"{r.get('start')}-{r.get('end')}")
                if self._range_matches_expected(rng, exp):
                    hits.append((key, r, exp))
            if len({h[0] for h in hits}) == 1:
                key, r, exp = hits[0]
                self.pending = {
                    "ply": key,
                    "expected_range": exp,
                    "bin": r.get("no_of_ply"),
                    "row": r,
                    "frame_index": sample.get("frame_index"),
                    "from_range_first": True,
                }
                return "pending"

        return None

    def status_line(self) -> str:
        if self.frozen:
            return (
                f"CONFIRMED ply={self.frozen.get('ply')} bin={self.frozen.get('bin')} "
                f"range={self.frozen.get('range')} [{self.frozen.get('reason')}]"
            )
        if self.pending:
            how = "need ply OCR" if self.pending.get("from_range_first") else "need start-end"
            return (
                f"PENDING ply={self.pending.get('ply')} expect={self.pending.get('expected_range')} "
                f"({how})"
            )
        return "idle: waiting for ply (then start-end confirm)"


def _put_latest(q: queue.Queue, item) -> None:
    """Keep only the newest job (drop stale frames if OCR is still busy)."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


def run_video(
    video_path: Path | str,
    output_dir: Path,
    master_path: Path,
    every_n: int,
    max_frames: int | None,
    max_side: int,
    save_empty: bool,
    live: bool = False,
    live_playback_fps: float | None = None,
    roi: str = "bottom_right",
    zoom: float = 2.5,
    vote_window: int = 10,
    vote_min: int = 3,
    freeze_on_master: bool = True,
    clear_after_absent: int = 5,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    ann_dir = output_dir / "annotated"
    frames_dir.mkdir(exist_ok=True)
    ann_dir.mkdir(exist_ok=True)

    master = load_master_list(master_path)
    print(f"Master list: {master_path} ({len(master)} rows)")
    print(f"Engine: RapidOCR ONNX (PP-OCR mobile) — Mac/Pi CPU parity, Hailo unused")
    video_src = str(video_path)
    is_url = video_src.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    is_v4l = video_src.startswith("/dev/video") or video_src.isdigit()
    is_stream = is_url or is_v4l
    print(f"Video: {video_src}" + (" [live stream]" if is_stream else ""))
    if live:
        print(
            "Live preview ON — smooth stream @ source FPS; OCR on background worker "
            "(press q to quit, space to pause/resume)"
        )
        print("  Tip: from SSH use DISPLAY=:0 if the Pi has a local desktop/HDMI.")

    engine = PaddleOnnxEngine()
    by_ply = master_index(master)
    voter = PlyRangeConfirmer(by_ply=by_ply)
    print("Confirm mode: ply-in-master -> wait for matching start-end -> hold until new ply")
    # Backend: V4L2 for /dev/video*, FFMPEG for RTSP/HTTP, default for files.
    if is_v4l:
        src_open: str | int = int(video_src) if video_src.isdigit() else video_src
        cap = cv2.VideoCapture(src_open, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src_open)
    elif is_url:
        cap = cv2.VideoCapture(video_src, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video/stream: {video_src}")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    play_fps = live_playback_fps or (src_fps if src_fps > 1 else 15.0)
    # Live cameras often report fps=0 / frame_count=0
    if is_stream and play_fps <= 1:
        play_fps = live_playback_fps or 15.0
    print(
        f"Stream: {width}x{height} @ {src_fps:.2f} fps, "
        f"~{total if total > 0 else 'live'} frames; OCR every {every_n}"
    )
    try:
        roi_xyxy = parse_roi_arg(roi, width, height)
    except ValueError as e:
        raise SystemExit(e) from e
    if roi_xyxy is None:
        roi_xyxy = (0, 0, width, height)
    print(f"ROI: {roi} → {roi_xyxy}  zoom={zoom}x (2-row ply / start-end check ON)")

    window = "Kitting PaddleOCR live"
    if live:
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, min(1280, width or 1280), min(720, height or 720))
        except cv2.error as e:
            raise SystemExit(
                "Live display needs OpenCV GUI + a display.\n"
                "  From SSH on Pi with HDMI: DISPLAY=:0 python ... --live\n"
                "  Or omit --live for headless OCR.\n"
                "  If headless OpenCV: pip uninstall -y opencv-python-headless && pip install opencv-python\n"
                f"Original error: {e}"
            ) from e

    results: list[dict] = []
    idx = 0
    sampled = 0
    t_all = time.perf_counter()
    paused = False
    stop_requested = False

    last_dets: list[dict] = []
    last_ply = None
    last_rng = None
    last_status = None
    last_two_row: dict | None = None
    last_ocr_fps: float | None = None
    last_ocr_frame = -1
    display_fps = 0.0
    t_prev_show = time.perf_counter()

    def apply_ocr_result(frame_bgr: np.ndarray, frame_idx: int, out: dict) -> None:
        nonlocal sampled, last_dets, last_ply, last_rng, last_status
        nonlocal last_two_row, last_ocr_fps, last_ocr_frame

        last_dets = out["detections"]
        last_ply = out["final"]["ply_no_final"]
        last_rng = out["final"]["start_end_final"]
        last_status = out["final"]["master_status"]
        last_two_row = out.get("assembled", {}).get("two_row_layout")
        inf = float(out["inference_s"] or 0)
        last_ocr_fps = (1.0 / inf) if inf > 0 else None
        last_ocr_frame = frame_idx

        useful = [d for d in last_dets if not looks_like_plant_label(d["text"])]
        has_signal = bool(useful) or bool(last_ply) or bool(last_rng)
        tr = last_two_row or {}

        ply_ocr = out["final"].get("ply_no_ocr")
        ply_for_vote = None
        if ply_ocr and by_ply.get(str(ply_ocr)):
            ply_for_vote = str(ply_ocr)
        elif last_ply and by_ply.get(str(last_ply)):
            ply_for_vote = str(last_ply)
        elif last_ply:
            ply_for_vote = str(last_ply)

        row = by_ply.get(str(ply_for_vote)) if ply_for_vote else None
        range_for_vote = out["final"].get("start_end_ocr") or last_rng
        sample = {
            "ply": ply_for_vote,
            "range": range_for_vote,
            "bin": (row or {}).get("no_of_ply") if row else None,
            "status": last_status,
            "frame_index": frame_idx,
            "row": row,
            "det_texts": [d["text"] for d in useful],
            "two_row_top": tr.get("top_row"),
        }
        event = voter.observe(sample)
        if event == "frozen":
            fr = voter.frozen or {}
            print(
                f"  *** CONFIRMED: PLY {fr.get('ply')} -> BIN {fr.get('bin')} "
                f"range={fr.get('range')} ({fr.get('reason')}) ***"
            )
        elif event == "pending":
            pe = voter.pending or {}
            print(
                f"  *** PENDING: ply={pe.get('ply')} expect range {pe.get('expected_range')} "
                f"({'need ply' if pe.get('from_range_first') else 'need start-end'}) ***"
            )
        elif event and str(event).startswith("switched:"):
            print(f"  *** SWITCHED to new ply ({event}) ***")
        elif event and str(event).startswith("cleared:"):
            print(f"  *** CLEARED: was ply {event.split(':', 1)[1]} ***")

        if has_signal or save_empty:
            stem = f"frame_{frame_idx:06d}"
            cv2.imwrite(str(frames_dir / f"{stem}.jpg"), frame_bgr)
            ann = draw_overlay(
                frame_bgr,
                last_dets,
                frame_idx=frame_idx,
                display_fps=display_fps,
                ocr_fps=last_ocr_fps,
                ply=last_ply,
                rng=last_rng,
                status=last_status,
                ocr_age_frames=0,
                roi=roi_xyxy,
                two_row=last_two_row,
                frozen=voter.frozen,
                vote_status=voter.status_line(),
            )
            cv2.imwrite(str(ann_dir / f"{stem}.jpg"), ann)
            rec = {
                "frame_index": frame_idx,
                "time_s": round(frame_idx / src_fps, 3) if src_fps > 0 else None,
                **out,
                "annotated": str(ann_dir / f"{stem}.jpg"),
            }
            results.append(rec)
            print(
                f"  [{sampled + 1}] frame={frame_idx}  dets={len(out['detections'])}  "
                f"ocr=({out['final']['ply_no_ocr']}, {out['final']['start_end_ocr']})  "
                f"final=({last_ply}, {last_rng})  {last_status}  "
                f"2row={'OK' if tr.get('ok') else 'NO'} "
                f"[{tr.get('top_row')}|{tr.get('bottom_row')}]  {out['inference_s']}s"
            )
            sampled += 1

    def drain_ocr_results(result_q: queue.Queue, block_ms: float = 0.0) -> None:
        deadline = time.perf_counter() + max(0.0, block_ms)
        while True:
            timeout = max(0.0, deadline - time.perf_counter()) if block_ms > 0 else 0.0
            try:
                if timeout > 0:
                    item = result_q.get(timeout=timeout)
                else:
                    item = result_q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            frame_idx, frame_bgr, out = item
            apply_ocr_result(frame_bgr, frame_idx, out)
            if max_frames is not None and sampled >= max_frames:
                break

    job_q: queue.Queue | None = None
    result_q: queue.Queue | None = None
    worker: threading.Thread | None = None

    if live:
        job_q = queue.Queue(maxsize=1)
        result_q = queue.Queue()
        stop_worker = threading.Event()

        def ocr_worker() -> None:
            while not stop_worker.is_set():
                try:
                    job = job_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if job is None:
                    job_q.task_done()
                    break
                frame_bgr, frame_idx = job
                try:
                    out = process_frame(
                        engine,
                        frame_bgr,
                        master,
                        max_side=max_side,
                        roi_xyxy=roi_xyxy,
                        zoom=zoom,
                    )
                    result_q.put((frame_idx, frame_bgr, out))
                except Exception as exc:  # noqa: BLE001 — keep stream alive
                    print(f"  [ocr worker] frame={frame_idx} error: {exc}")
                finally:
                    job_q.task_done()

        worker = threading.Thread(target=ocr_worker, name="ocr-worker", daemon=True)
        worker.start()
        print(f"Live: display @ {play_fps:.1f} fps; OCR worker every {every_n} (non-blocking)")

    while True:
        if live and paused:
            assert result_q is not None
            drain_ocr_results(result_q)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                stop_requested = True
                break
            if key == ord(" "):
                paused = False
            continue

        ok, frame = cap.read()
        if not ok:
            break

        if live:
            assert job_q is not None and result_q is not None
            drain_ocr_results(result_q)
            can_submit = max_frames is None or sampled < max_frames
            if can_submit and idx % every_n == 0:
                # Copy so capture buffer can advance while OCR runs.
                _put_latest(job_q, (frame.copy(), idx))
        else:
            if idx % every_n == 0:
                out = process_frame(
                    engine, frame, master, max_side=max_side, roi_xyxy=roi_xyxy, zoom=zoom
                )
                apply_ocr_result(frame, idx, out)

        now = time.perf_counter()
        dt_show = now - t_prev_show
        if dt_show > 0:
            instant = 1.0 / dt_show
            display_fps = 0.9 * display_fps + 0.1 * instant if display_fps > 0 else instant
        t_prev_show = now

        if live:
            age = idx - last_ocr_frame if last_ocr_frame >= 0 else 0
            vis = draw_overlay(
                frame,
                last_dets,
                frame_idx=idx,
                display_fps=display_fps,
                ocr_fps=last_ocr_fps,
                ply=last_ply,
                rng=last_rng,
                status=last_status,
                ocr_age_frames=age,
                roi=roi_xyxy,
                two_row=last_two_row,
                frozen=voter.frozen,
                vote_status=voter.status_line(),
            )
            cv2.imshow(window, vis)
            # File playback paces to play_fps; live RTSP shows ASAP (less backlog lag).
            delay_ms = 1 if is_stream else max(1, int(1000 / play_fps))
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):
                stop_requested = True
                break
            if key == ord(" "):
                paused = True

        idx += 1
        if max_frames is not None and sampled >= max_frames:
            break

    # Drain in-flight OCR before tearing down (live path).
    if live and job_q is not None and result_q is not None and worker is not None:
        if not stop_requested:
            # Wait for the current OCR job so end-of-video samples aren't lost.
            for _ in range(80):
                drain_ocr_results(result_q, block_ms=0.05)
                if job_q.empty() and result_q.empty():
                    break
        stop_worker.set()
        _put_latest(job_q, None)
        worker.join(timeout=8.0)
        drain_ocr_results(result_q)

    cap.release()
    if live:
        cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t_all

    ok_frames = [
        r
        for r in results
        if r["final"]["master_status"]
        in ("corrected_via_4to9_9to7_and_master", "ply_found_exact_ocr")
    ]
    summary = {
        "video": video_src,
        "is_stream": is_stream,
        "master_list": str(master_path),
        "engine": "rapidocr-onnx-ppocr",
        "parity_note": (
            "ONNX PP-OCR on CPU — same models on Mac and Pi 5. "
            "Hailo AI HAT not used (would change accuracy vs Mac)."
        ),
        "pi_note": "Pi 5 2GB: --live uses a background OCR worker; --every-n 10+ and --max-side 960 recommended.",
        "live": live,
        "live_async_ocr": live,
        "video_meta": {
            "width": width,
            "height": height,
            "fps": src_fps,
            "frame_count": total,
            "every_n": every_n,
        },
        "sampled_saved": len(results),
        "master_ok_count": len(ok_frames),
        "frozen": voter.frozen,
        "vote_window": vote_window,
        "vote_min": vote_min,
        "elapsed_s": round(elapsed, 2),
        "working_verdict": (
            "YES — at least one frame matched master (exact or confusion)"
            if ok_frames
            else "NO / UNCLEAR — no frame matched master; inspect annotated/ and raw OCR"
        ),
        "frames": results,
    }

    json_path = output_dir / "paddle_video_result.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== VIDEO OCR SUMMARY =====")
    print(f"  Engine:     {summary['engine']} (Mac/Pi ONNX parity)")
    print(f"  Sampled:    {len(results)} frames (every {every_n})")
    print(f"  Master OK:  {len(ok_frames)}")
    print(f"  Verdict:    {summary['working_verdict']}")
    if voter.frozen:
        print(
            f"  FROZEN:     PLY {voter.frozen.get('ply')} → BIN {voter.frozen.get('bin')} "
            f"({voter.frozen.get('reason')})"
        )
    print(f"  Wrote:     {json_path}")
    print(f"  Annotated:  {ann_dir}")
    return summary


def main():
    root = Path(__file__).resolve().parents[1]
    default_video = Path(
        "/Users/kanishka/Downloads/kitting_sept_12/lm_cpe_2026-09-13_05-18-22/cam0.mp4"
    )
    p = argparse.ArgumentParser(description="Paddle/ONNX PP-OCR video test (Mac ↔ Pi parity)")
    p.add_argument(
        "--video",
        type=str,
        default=str(default_video),
        help="Path to video file, /dev/videoN, camera index (0), or RTSP/HTTP URL",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output" / "paddle_video_cam0",
    )
    p.add_argument(
        "--master-list",
        type=Path,
        default=root / "data" / "master_list.csv",
    )
    p.add_argument(
        "--every-n",
        type=int,
        default=30,
        help="Run OCR every Nth frame (boxes/text persist between OCR ticks)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many OCR samples (smoke test)",
    )
    p.add_argument(
        "--max-side",
        type=int,
        default=1280,
        help="Resize longer side before OCR (use 960 on Pi 2GB)",
    )
    p.add_argument(
        "--save-empty",
        action="store_true",
        help="Also save frames with zero detections",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Show live video window with FPS + detected/recognised text overlay",
    )
    p.add_argument(
        "--live-fps",
        type=float,
        default=None,
        help="Playback pace for live view (default: source FPS). OCR runs on a worker and does not hitch the stream.",
    )
    p.add_argument(
        "--roi",
        type=str,
        default="bottom_right",
        help="Crop focus: bottom_right|lower_half|right_half|write_desk|full or x1,y1,x2,y2 fractions",
    )
    p.add_argument(
        "--zoom",
        type=float,
        default=2.5,
        help="Upscale ROI before OCR (helps small handwriting on overhead cams)",
    )
    p.add_argument(
        "--vote-window",
        type=int,
        default=10,
        help="Recent OCR samples used for ply voting (e.g. 10)",
    )
    p.add_argument(
        "--vote-min",
        type=int,
        default=3,
        help="Freeze when this many samples in the window agree on the same ply",
    )
    p.add_argument(
        "--no-freeze-on-master",
        action="store_true",
        help="Do not freeze immediately on first master match (vote only)",
    )
    p.add_argument(
        "--clear-after-absent",
        type=int,
        default=5,
        help="Clear frozen PLY/BIN overlay after this many OCR ticks without that ply in ROI",
    )
    args = p.parse_args()

    video_arg = args.video.strip()
    is_url = video_arg.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    is_v4l = video_arg.startswith("/dev/video") or video_arg.isdigit()
    if not is_url and not is_v4l and not Path(video_arg).exists():
        raise SystemExit(f"Video not found: {video_arg}")
    if is_v4l and video_arg.startswith("/dev/video") and not Path(video_arg).exists():
        raise SystemExit(f"Camera device not found: {video_arg}")

    run_video(
        video_path=video_arg,
        output_dir=args.output_dir,
        master_path=args.master_list,
        every_n=args.every_n,
        max_frames=args.max_frames,
        max_side=args.max_side,
        save_empty=args.save_empty,
        live=args.live,
        live_playback_fps=args.live_fps,
        roi=args.roi,
        zoom=args.zoom,
        vote_window=args.vote_window,
        vote_min=args.vote_min,
        freeze_on_master=not args.no_freeze_on_master,
        clear_after_absent=args.clear_after_absent,
    )


if __name__ == "__main__":
    main()