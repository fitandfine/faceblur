#!/usr/bin/env python3
"""
Real-time Face Blur with GPU Acceleration + Virtual Webcam

Features:
- Detect faces with facenet-pytorch MTCNN (GPU if available).
- Blur faces with Gaussian blur (torch on CUDA, else OpenCV CPU).
- Interactive controls:
    R = toggle recording
    S = save snapshot
    B = increase blur
    N = decrease blur
    Q or ESC = quit
- Overlay UI instructions on video.
- Optional virtual webcam output (Linux: v4l2loopback, all OS: pyvirtualcam).

Dependencies:
    pip install opencv-python torch facenet-pytorch numpy pyvirtualcam
"""

import argparse
import os
from datetime import datetime

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None

try:
    import pyvirtualcam
except ImportError:
    pyvirtualcam = None


# ---------- Gaussian Blur (GPU + CPU) ----------

def make_gaussian_kernel2d(kernel_size: int, sigma: float, device):
    """Builds a 2D Gaussian kernel for torch conv2d."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    x, y = coords.view(1, -1), coords.view(-1, 1)
    g = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    g /= g.sum()
    return g.view(1, 1, kernel_size, kernel_size)


def blur_roi_torch(roi, kernel_size, sigma, device):
    """Blur ROI using torch conv2d (fast on GPU)."""
    tensor = torch.from_numpy(roi.astype(np.float32) / 255.0).to(device)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
    C = tensor.shape[1]
    kernel = make_gaussian_kernel2d(kernel_size, sigma, device).repeat(C, 1, 1, 1)
    pad = kernel_size // 2
    tensor = torch.nn.functional.pad(tensor, (pad, pad, pad, pad), mode="reflect")
    blurred = torch.nn.functional.conv2d(tensor, kernel, groups=C)
    blurred = (blurred.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return blurred


def blur_roi_cv2(roi, kernel_size, sigma):
    """Blur ROI using OpenCV (CPU)."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(roi, (kernel_size, kernel_size), sigma)


# ---------- Main App ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Video source (0=webcam, or path to video)")
    parser.add_argument("--save-dir", default="captures", help="Save dir for video/snapshots")
    parser.add_argument("--blur-ksize", type=int, default=41, help="Initial blur kernel size")
    parser.add_argument("--blur-sigma", type=float, default=12.0, help="Initial blur sigma")
    parser.add_argument("--use-virtualcam", action="store_true", help="Output to virtual webcam (needs pyvirtualcam)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Device + detector
    device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=str(device)) if MTCNN else None

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    print(f"[INFO] Running on {device}, source={args.source}, {width}x{height}@{fps:.1f}fps")

    # States
    blur_ksize, blur_sigma = args.blur_ksize, args.blur_sigma
    is_recording, writer = False, None
    use_cuda_blur = torch and device.type == "cuda"

    # VirtualCam
    cam = None
    if args.use_virtualcam and pyvirtualcam:
        cam = pyvirtualcam.Camera(width, height, fps=fps)
        print(f"[INFO] Virtual camera started: {cam.device}")

    cv2.namedWindow("FaceBlur", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb) if mtcnn else (None, None)

        if boxes is not None:
            for (x1, y1, x2, y2) in boxes.astype(int):
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                try:
                    if use_cuda_blur:
                        roi_blurred = blur_roi_torch(roi, blur_ksize, blur_sigma, device)
                    else:
                        roi_blurred = blur_roi_cv2(roi, blur_ksize, blur_sigma)
                    frame[y1:y2, x1:x2] = roi_blurred
                except Exception:
                    frame[y1:y2, x1:x2] = blur_roi_cv2(roi, blur_ksize, blur_sigma)

        # Overlay help text
        overlay = [
            f"[B/N] Blur +/- (Current: {blur_ksize}, sigma={blur_sigma})",
            "[R]ecord toggle | [S]napshot | [Q]uit",
            f"Recording: {'ON' if is_recording else 'OFF'}"
        ]
        y0 = 25
        for i, text in enumerate(overlay):
            cv2.putText(frame, text, (10, y0 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Show or send to virtual cam
        if cam:
            cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cam.sleep_until_next_frame()
        else:
            cv2.imshow("FaceBlur", frame)

        # Handle recording
        if is_recording and writer:
            writer.write(frame)

        # Key handling
        k = cv2.waitKey(1) & 0xFF
        if k in [ord("q"), 27]:
            break
        elif k == ord("r"):
            is_recording = not is_recording
            if is_recording:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(args.save_dir, f"output_{ts}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                print(f"[INFO] Recording to {out_path}")
            else:
                writer.release(); writer = None
                print("[INFO] Recording stopped")
        elif k == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(args.save_dir, f"snap_{ts}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"[INFO] Snapshot saved {img_path}")
        elif k == ord("b"):  # increase blur
            blur_ksize = min(blur_ksize+2, 101)
        elif k == ord("n"):  # decrease blur
            blur_ksize = max(3, blur_ksize-2)

    cap.release()
    if writer: writer.release()
    if cam: cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
