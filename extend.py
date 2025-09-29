#!/usr/bin/env python3
"""
anpr_qr_ocr.py

Real-time capture that:
 - optionally detects faces and blurs them (facenet-pytorch MTCNN)
 - detects QR codes & barcodes (pyzbar), decodes them, saves snapshots + JSON records
 - extracts text via OCR (pytesseract and optional easyocr)
 - attempts to detect license-plate-like regions (contour heuristics + OCR)
 - outputs records to JSONL file (one JSON object per line)

Usage:
    python anpr_qr_ocr.py --source 0 --outdir captures --json db.jsonl --no-blur

Notes:
 - Requires system packages: tesseract-ocr and libzbar (installation examples in README notes).
 - This script is intentionally defensive and logs errors rather than crashing.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any

import cv2
import numpy as np

# barcode/QR decoder
from pyzbar import pyzbar

# OCR
import pytesseract
# optional: fallback/alternative OCR (works well for some plates)
try:
    import easyocr
except Exception:
    easyocr = None

# optional GPU face detector
try:
    import torch
    from facenet_pytorch import MTCNN
except Exception:
    torch = None
    MTCNN = None

# ---------------- utilities ----------------

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_image(img: np.ndarray, outdir: str, prefix: str) -> str:
    fn = f"{prefix}_{timestamp()}.jpg"
    path = os.path.join(outdir, fn)
    cv2.imwrite(path, img)
    return path

def append_json_record(jsonl_path: str, record: Dict[str, Any]):
    with open(jsonl_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

# ---------------- barcode / QR helpers ----------------

def decode_barcodes(frame_gray: np.ndarray):
    """Use pyzbar to detect and decode barcodes/QR codes. Returns list of dicts."""
    decoded = pyzbar.decode(frame_gray)
    results = []
    for d in decoded:
        # d.rect contains left, top, width, height
        x, y, w, h = d.rect.left, d.rect.top, d.rect.width, d.rect.height
        data = d.data.decode("utf-8", errors="replace")
        typ = d.type
        results.append({"type": typ, "data": data, "bbox": (x, y, w, h)})
    return results

# ---------------- OCR helpers ----------------

def ocr_tesseract(img: np.ndarray, lang: str = "eng") -> str:
    """Run pytesseract OCR on a BGR or grayscale image and return text."""
    try:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        # basic threshold for better contrast (optional)
        # _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        txt = pytesseract.image_to_string(gray, lang=lang, config="--psm 6")
        return txt.strip()
    except Exception as e:
        print("Tesseract OCR failed:", e)
        return ""

def ocr_easyocr(img: np.ndarray, reader=None) -> str:
    """Use easyocr if available (slower but often robust)."""
    if easyocr is None:
        return ""
    if reader is None:
        # create reader for English (fast: model is cached)
        reader = easyocr.Reader(["en"], gpu=torch and torch.cuda.is_available())
    try:
        results = reader.readtext(img)
        # join text segments
        texts = [t[1] for t in results]
        return " ".join(texts)
    except Exception as e:
        print("easyOCR failure:", e)
        return ""

# ---------------- simple plate candidate detection ----------------

def find_plate_candidates(frame_gray: np.ndarray, min_area=1000, max_area=30000):
    """
    Heuristic: find rectangular contours likely to be license plates.
    Returns list of bbox (x,y,w,h).
    This is a heuristic — accuracy varies by camera, environment, plate shapes.
    """
    # edge detection
    blurred = cv2.GaussianBlur(frame_gray, (5,5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    # morphological close to join edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x,y,w,h = cv2.boundingRect(c)
        ar = w / float(h) if h>0 else 0
        # license plates usually have 2:5 to 1:4 aspect ratio depending on region
        if 2.0 < ar < 6.0:
            candidates.append((x,y,w,h))
    return candidates

def is_plate_text(text: str) -> bool:
    """Very simple heuristic to decide if OCR text looks like a plate string."""
    if not text:
        return False
    t = text.replace(" ", "").replace("\n", "")
    # keep alnum characters
    t2 = "".join(ch for ch in t if ch.isalnum())
    if len(t2) < 4 or len(t2) > 10:
        return False
    # require at least one digit
    if not any(ch.isdigit() for ch in t2):
        return False
    # success
    return True

# ---------------- main processing loop ----------------

def main():
    parser = argparse.ArgumentParser(description="Real-time barcode/QR/OCR + optional face-blur + plate heuristic")
    parser.add_argument("--source", default="0", help="video source (0 or path)")
    parser.add_argument("--outdir", default="captures", help="folder to save snapshots")
    parser.add_argument("--json", default="detections.jsonl", help="JSONL output file")
    parser.add_argument("--blur-faces", action="store_true", help="blur faces (requires facenet-pytorch)")
    parser.add_argument("--use-easyocr", action="store_true", help="use easyocr in addition to tesseract (may be slower)")
    parser.add_argument("--min-plate-area", type=int, default=1000)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    jsonl_path = args.json

    # video capture
    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Failed to open source:", src)
        sys.exit(1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[INFO] source={src} size={width}x{height} @ {fps:.1f}fps")

    # face detector
    device = None
    mtcnn = None
    if args.blur_faces and MTCNN is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mtcnn = MTCNN(keep_all=True, device=str(device))
        print("[INFO] MTCNN ready on", device)
    else:
        if args.blur_faces:
            print("[WARN] blur faces requested but facenet-pytorch not available; continuing without face blur")

    easy_reader = None
    if args.use_easyocr and easyocr is not None:
        easy_reader = easyocr.Reader(["en"], gpu=torch and torch.cuda.is_available())

    # keep track of recent detections to avoid repeated saves
    recent_codes = {}  # map(data -> last_seen_time)
    dedupe_seconds = 5.0

    cv2.namedWindow("ANPR-QR-OCR", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        orig = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) detect + blur faces (optional)
        if mtcnn:
            try:
                # mtcnn returns boxes in (x1,y1,x2,y2)
                boxes, _ = mtcnn.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if boxes is not None:
                    for b in boxes.astype(int):
                        x1,y1,x2,y2 = b
                        # clamp
                        x1, y1 = max(x1,0), max(y1,0)
                        x2, y2 = min(x2, frame.shape[1]-1), min(y2, frame.shape[0]-1)
                        roi = frame[y1:y2, x1:x2]
                        if roi.size==0: continue
                        try:
                            # quick blur on CPU for simplicity (could use GPU conv)
                            frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51,51), 0)
                        except Exception:
                            pass
            except Exception as e:
                print("Face detection error:", e)

        # 2) detect barcodes / QR
        try:
            bar_results = decode_barcodes(gray)
            for r in bar_results:
                x,y,w,h = r["bbox"]
                # draw
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                label = f"{r['type']}: {r['data']}"
                cv2.putText(frame, label, (x, max(y-10,10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # dedupe
                now = time.time()
                last = recent_codes.get(r['data'])
                if last and (now - last) < dedupe_seconds:
                    continue
                recent_codes[r['data']] = now

                # save snapshot of ROI and record JSON
                x0,y0 = max(x,0), max(y,0)
                roi_img = orig[y0:y0+h, x0:x0+w]
                saved = save_image(roi_img, args.outdir, prefix=f"code_{r['type']}")
                record = {
                    "ts": datetime.utcnow().isoformat(),
                    "kind": "barcode" if r['type'] != "QRCODE" else "qrcode",
                    "format": r['type'],
                    "data": r['data'],
                    "bbox": [int(x),int(y),int(w),int(h)],
                    "image": saved
                }
                append_json_record(jsonl_path, record)
                print("[DETECTED]", record['kind'], record['data'], "->", saved)
        except Exception as e:
            print("Barcode decode error:", e)

        # 3) OCR for textual regions (we'll run OCR on barcode ROI and plate candidates)
        # 3a) simple: run OCR over full frame at low frequency? (costly). We instead run on candidates.
        # Plate candidates:
        try:
            candidates = find_plate_candidates(gray, min_area=args.min_plate_area)
            for (x,y,w,h) in candidates:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
                roi = orig[y:y+h, x:x+w]
                # try tesseract
                text = ocr_tesseract(roi)
                # optionally easyocr
                if args.use_easyocr and easy_reader:
                    e_text = ocr_easyocr(roi, reader=easy_reader)
                    if len(e_text) > len(text):
                        text = e_text
                if is_plate_text(text):
                    key = f"PLATE::{text}"
                    now = time.time()
                    last = recent_codes.get(key)
                    if last and (now - last) < dedupe_seconds:
                        continue
                    recent_codes[key] = now
                    saved = save_image(roi, args.outdir, prefix="plate")
                    record = {
                        "ts": datetime.utcnow().isoformat(),
                        "kind": "plate_candidate",
                        "text": text,
                        "bbox": [int(x),int(y),int(w),int(h)],
                        "image": saved
                    }
                    append_json_record(jsonl_path, record)
                    print("[PLATE CAND]", text, "->", saved)
                else:
                    # optionally label with small text
                    cv2.putText(frame, "plate?", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        except Exception as e:
            print("Plate detection error:", e)

        # 3b) OCR on barcode ROI text (sometimes QR contains text not machine-readable)
        # (pyzbar already gave us decoded text; if you want OCR on the code image as backup, do that)
        # We'll optionally run OCR over the whole frame every N frames — omitted here for speed.

        # 4) show UI with counts and status (minimal overlay)
        cv2.putText(frame, f"QR/Barcodes detected (last {dedupe_seconds}s): {len([k for k,t in recent_codes.items() if time.time()-t < dedupe_seconds])}",
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("ANPR-QR-OCR", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("s"):
            p = save_image(orig, args.outdir, prefix="frame")
            print("[SNAPSHOT]", p)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] exiting")

if __name__ == "__main__":
    main()
