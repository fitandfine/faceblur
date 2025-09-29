#!/usr/bin/env python3
"""
Real-time Face Blur with GPU Acceleration + GUI controls (Tkinter)

Changes / improvements made:
- Replaced the blocking cv2.imshow loop with a Tkinter UI that:
  * Displays the video in a larger pane (configurable display_scale).
  * Moves the "help / control guide" out of the video and into a side control panel.
  * Provides four arrow buttons (and arrow-key bindings) to PAN the visible area
    when zoomed in. Pan logic works by cropping a region from the processed frame
    and resizing that crop to the display area (zoom effect).
  * Adds a "Properties" button which opens a properties window (Toplevel) with
    runtime info and live sliders for blur kernel size and sigma.
  * Keeps original functionality: R / S / B / N / Q keys still work.
- Virtual webcam (pyvirtualcam) behavior unchanged: virtual camera still receives
  the full processed frame (not the cropped display frame), so recording/virtual
  webcam output preserves original resolution.
- All original face blur logic remains intact; GPU torch blur is still used if available.
- Robust clamping and odd-kernel enforcement for Gaussian blur size.

Dependencies:
  pip install pillow opencv-python facenet-pytorch torch pyvirtualcam numpy

Run:
  python face_blur_gui.py --use-virtualcam   # or without the flag

Author: adapted for you
"""

import argparse
import os
from datetime import datetime
import sys
import traceback

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

# Pillow (for converting cv2 frames to Tkinter PhotoImage)
try:
    from PIL import Image, ImageTk
except Exception:
    print("Missing dependency: Pillow is required for the Tkinter UI display.")
    print("Install with: pip install pillow")
    raise

# Optional accelerated libraries
try:
    import torch
except Exception:
    torch = None

try:
    from facenet_pytorch import MTCNN
except Exception:
    MTCNN = None

try:
    import pyvirtualcam
except Exception:
    pyvirtualcam = None


# ---------- Gaussian Blur (GPU + CPU) ----------
def make_gaussian_kernel2d(kernel_size: int, sigma: float, device):
    """Build a 2D Gaussian kernel suitable for conv2d on torch tensors.

    Returns shape (1,1,ks,ks). Caller repeats across channels.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    x = coords.view(1, -1)
    y = coords.view(-1, 1)
    g = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.view(1, 1, kernel_size, kernel_size)


def blur_roi_torch(roi, kernel_size, sigma, device):
    """Blur ROI using torch conv2d — best on CUDA device."""
    # roi: HxWxC (uint8)
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
    """Blur ROI using OpenCV (CPU fallback)."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(roi, (kernel_size, kernel_size), sigma)


# ---------- Helper utilities ----------
def ensure_odd(n: int):
    """Ensure returned int is odd and >= 3."""
    n = max(3, int(n))
    if n % 2 == 0:
        n += 1
    return n


# ---------- Main Application (Tkinter based) ----------
class FaceBlurApp:
    def __init__(self, args):
        self.args = args
        os.makedirs(args.save_dir, exist_ok=True)

        # Capture
        self.cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Unable to open video source: {args.source}")
            raise RuntimeError(f"Cannot open source {args.source}")

        # original frame dims
        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)

        # Torch + detector
        self.device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=True, device=str(self.device)) if MTCNN else None

        # Blur parameters
        self.blur_ksize = ensure_odd(args.blur_ksize)
        self.blur_sigma = float(args.blur_sigma)
        self.use_cuda_blur = (torch is not None) and (self.device.type == "cuda")

        # Recording state
        self.is_recording = False
        self.writer = None

        # Virtualcam (receives full processed frame)
        self.cam = None
        if args.use_virtualcam:
            if pyvirtualcam is None:
                print("[WARN] pyvirtualcam requested but not installed.")
            else:
                # create virtual cam with original capture resolution so sending frames is straightforward
                try:
                    self.cam = pyvirtualcam.Camera(self.orig_w, self.orig_h, fps=self.fps)
                    print("[INFO] Virtual camera started:", self.cam.device)
                except Exception as e:
                    print("[WARN] Failed to start virtual camera:", e)
                    self.cam = None

        # GUI parameters
        self.display_scale = 1.5  # how much bigger the display area should be (1.0 = original)
        # derived display canvas size
        self.display_w = min(1280, int(self.orig_w * self.display_scale))
        self.display_h = int(self.display_w * self.orig_h / self.orig_w)

        # Zoom and pan
        self.zoom_factor = 1.0  # 1.0 = no zoom; >1 = zoom in (thus enabling pan)
        self.pan_x = 0
        self.pan_y = 0

        # Tk root and UI elements (set up in run_ui)
        self.root = None
        self.video_label = None
        self.prop_window = None
        self.prop_widgets = {}  # references for updating

        # Running flag for after-loop
        self.running = True

        # Last displayed PhotoImage ref (prevent GC)
        self._photoimage = None

    # ---------------- GUI / interaction ----------------
    def run_ui(self):
        """Initialize Tkinter UI and start the update loop (after-based)."""
        self.root = tk.Tk()
        self.root.title("FaceBlur - Real-time face blurring (GUI)")

        # main layout: left video pane, right controls pane
        main_fr = ttk.Frame(self.root)
        main_fr.pack(fill="both", expand=True)

        # Video frame
        video_fr = ttk.Frame(main_fr)
        video_fr.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        self.video_label = tk.Label(video_fr, bd=2, relief="sunken")
        self.video_label.pack(fill="both", expand=True)

        # Controls frame
        ctrl_fr = ttk.Frame(main_fr, width=300)
        ctrl_fr.pack(side="right", fill="y", padx=6, pady=6)

        # Top controls: Blur + Snapshot + Record
        ttk.Label(ctrl_fr, text="Controls", font=("Segoe UI", 12, "bold")).pack(pady=(0,8))

        top_buttons = ttk.Frame(ctrl_fr)
        top_buttons.pack(fill="x", pady=4)
        ttk.Button(top_buttons, text="Snapshot (S)", command=self.take_snapshot).pack(side="left", expand=True, padx=2)
        ttk.Button(top_buttons, text="Record (R)", command=self.toggle_recording).pack(side="left", expand=True, padx=2)

        # Blur controls
        blur_fr = ttk.LabelFrame(ctrl_fr, text="Blur settings", padding=6)
        blur_fr.pack(fill="x", pady=8)
        ttk.Label(blur_fr, text="Kernel size (odd)").pack(anchor="w")
        self.blk_slider = tk.Scale(blur_fr, from_=3, to=101, orient="horizontal",
                                   resolution=2, command=self._on_blur_ksize_change)
        self.blk_slider.set(self.blur_ksize)
        self.blk_slider.pack(fill="x", pady=4)
        ttk.Label(blur_fr, text="Sigma").pack(anchor="w")
        self.sigma_slider = tk.Scale(blur_fr, from_=0.5, to=50.0, orient="horizontal",
                                     resolution=0.5, command=self._on_blur_sigma_change)
        self.sigma_slider.set(self.blur_sigma)
        self.sigma_slider.pack(fill="x", pady=4)

        # Pan controls
        pan_fr = ttk.LabelFrame(ctrl_fr, text="Pan (when zoomed)", padding=6)
        pan_fr.pack(fill="x", pady=8)
        # Arrow buttons laid out in grid
        btn_up = ttk.Button(pan_fr, text="↑", width=4, command=lambda: self.pan_by(0, -1))
        btn_left = ttk.Button(pan_fr, text="←", width=4, command=lambda: self.pan_by(-1, 0))
        btn_right = ttk.Button(pan_fr, text="→", width=4, command=lambda: self.pan_by(1, 0))
        btn_down = ttk.Button(pan_fr, text="↓", width=4, command=lambda: self.pan_by(0, 1))
        btn_reset = ttk.Button(pan_fr, text="Reset Pan", command=self.reset_pan)
        btn_up.grid(row=0, column=1, padx=4, pady=2)
        btn_left.grid(row=1, column=0, padx=4, pady=2)
        btn_right.grid(row=1, column=2, padx=4, pady=2)
        btn_down.grid(row=2, column=1, padx=4, pady=2)
        btn_reset.grid(row=3, column=0, columnspan=3, pady=(6,0), sticky="we")

        # Properties button
        props_fr = ttk.Frame(ctrl_fr)
        props_fr.pack(fill="x", pady=(8, 2))
        ttk.Button(props_fr, text="Properties", command=self.open_properties_window).pack(fill="x")

        # Status panel at bottom
        status_fr = ttk.LabelFrame(ctrl_fr, text="Status", padding=6)
        status_fr.pack(fill="both", expand=True, pady=(8, 0))
        self.status_text = tk.StringVar()
        self.status_text.set(self._status_text())
        ttk.Label(status_fr, textvariable=self.status_text, justify="left").pack(anchor="nw")

        # Key bindings: keep keyboard controls convenient
        self.root.bind("<Key>", self._on_key)
        self.root.bind("<Left>", lambda e: self.pan_by(-1, 0))
        self.root.bind("<Right>", lambda e: self.pan_by(1, 0))
        self.root.bind("<Up>", lambda e: self.pan_by(0, -1))
        self.root.bind("<Down>", lambda e: self.pan_by(0, 1))

        # On close -> cleanup
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start the update loop (reads frames & updates UI)
        # Use after scheduling at approx fps interval
        interval_ms = int(1000 / max(1.0, self.fps))
        self._after_id = None
        self._schedule_next_frame(interval_ms)

        # Start Tk mainloop
        self.root.mainloop()

    def _schedule_next_frame(self, interval_ms):
        """Schedule the frame update periodically using Tk 'after'."""
        if not self.running or not self.root:
            return
        # use lambda so we can recompute interval if needed
        self._after_id = self.root.after(interval_ms, self._update_frame_and_ui)

    # ---------------- frame processing ----------------
    def _update_frame_and_ui(self):
        """Read a frame, process faces, update virtual cam (if any), and paint the Tk label."""
        try:
            ret, frame = self.cap.read()
            if not ret:
                # end of video file or camera lost: stop gracefully
                print("[INFO] Video source ended or cannot read frame.")
                self._on_close()
                return

            # Make a working copy (we blur in-place into this)
            processed = frame.copy()

            # Detect faces and blur them
            try:
                rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                boxes, _ = self.mtcnn.detect(rgb) if self.mtcnn else (None, None)
            except Exception:
                # In case detector raises: log and continue (no faces)
                print("[WARN] Detector failed:", traceback.format_exc())
                boxes = None

            if boxes is not None:
                for (x1, y1, x2, y2) in boxes.astype(int):
                    # clamp coords
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(self.orig_w, x2), min(self.orig_h, y2)
                    roi = processed[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    # Try GPU blur first (if enabled), else cv2
                    try:
                        if self.use_cuda_blur:
                            roi_blurred = blur_roi_torch(roi, self.blur_ksize, self.blur_sigma, self.device)
                        else:
                            roi_blurred = blur_roi_cv2(roi, self.blur_ksize, self.blur_sigma)
                        processed[y1:y2, x1:x2] = roi_blurred
                    except Exception:
                        # On any failure, fallback to cv2 CPU blur
                        processed[y1:y2, x1:x2] = blur_roi_cv2(roi, self.blur_ksize, self.blur_sigma)

            # Virtual cam receives full processed frame (original resolution)
            if self.cam:
                try:
                    # pyvirtualcam expects RGB frames as ndarray
                    self.cam.send(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
                    self.cam.sleep_until_next_frame()
                except Exception:
                    # Non-fatal: print and continue
                    print("[WARN] virtual cam send failed:", traceback.format_exc())

            # If recording, write original processed frame (orig resolution)
            if self.is_recording and self.writer is not None:
                self.writer.write(processed)

            # Crop & resize for display depending on zoom_factor and pan
            display_img = self._crop_and_resize_for_display(processed)

            # Convert BGR->RGB and paint on Tk label via PIL ImageTk
            rgb_for_tk = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_for_tk)
            # Resize PIL to exact display canvas if necessary (should already be right)
            if img_pil.size != (self.display_w, self.display_h):
                img_pil = img_pil.resize((self.display_w, self.display_h), Image.BILINEAR)

            self._photoimage = ImageTk.PhotoImage(image=img_pil)
            self.video_label.configure(image=self._photoimage)

            # update status text
            self.status_text.set(self._status_text())

        except Exception:
            print("[ERROR] Frame update failed:", traceback.format_exc())

        # schedule next frame
        interval_ms = int(1000 / max(1.0, self.fps))
        self._schedule_next_frame(interval_ms)

    def _crop_and_resize_for_display(self, frame):
        """Crop the processed frame according to zoom_factor and pan, then resize for display.

        Strategy:
         - If zoom_factor == 1.0: just resize original frame to display canvas
         - If zoom_factor > 1.0: calculate a smaller crop region (orig_w/zoom, orig_h/zoom),
           use pan offsets to pick the region, then resize that crop to the display canvas.
        """
        h, w = frame.shape[:2]

        # clamp zoom
        zoom = max(1.0, float(self.zoom_factor))
        if zoom <= 1.0:
            # simply scale to display canvas
            if (self.display_w, self.display_h) != (w, h):
                return cv2.resize(frame, (self.display_w, self.display_h), interpolation=cv2.INTER_LINEAR)
            return frame

        # compute source crop size
        crop_w = max(1, int(w / zoom))
        crop_h = max(1, int(h / zoom))

        # clamp pan offsets so crop remains inside frame
        max_x = max(0, w - crop_w)
        max_y = max(0, h - crop_h)
        self.pan_x = int(max(0, min(self.pan_x, max_x)))
        self.pan_y = int(max(0, min(self.pan_y, max_y)))

        x0, y0 = self.pan_x, self.pan_y
        x1, y1 = x0 + crop_w, y0 + crop_h

        # crop and resize to display size
        crop = frame[y0:y1, x0:x1]
        disp = cv2.resize(crop, (self.display_w, self.display_h), interpolation=cv2.INTER_LINEAR)
        return disp

    # ---------------- UI actions ----------------
    def _on_key(self, event):
        """Key binding handler for R/S/B/N/Q etc."""
        key = event.keysym.lower()
        if key == "q" or event.keycode == 27:
            self._on_close()
        elif key == "r":
            self.toggle_recording()
        elif key == "s":
            self.take_snapshot()
        elif key == "b":
            # Increase blur kernel size by 2, maintain oddness
            self.blur_ksize = ensure_odd(min(101, self.blur_ksize + 2))
            self.blk_slider.set(self.blur_ksize)
        elif key == "n":
            self.blur_ksize = ensure_odd(max(3, self.blur_ksize - 2))
            self.blk_slider.set(self.blur_ksize)
        elif key in ("left", "right", "up", "down"):
            # arrow keys already bound separately
            pass

    def pan_by(self, dx_unit: int, dy_unit: int):
        """Pan relative to the crop size.

        dx_unit/dy_unit are step multipliers (e.g., -1, 0, 1). We compute a dynamic step
        (fraction of current crop size) so the pan feels natural regardless of zoom.
        """
        if self.zoom_factor <= 1.0:
            # nothing to pan
            return

        # compute current crop size
        crop_w = max(1, int(self.orig_w / self.zoom_factor))
        crop_h = max(1, int(self.orig_h / self.zoom_factor))

        # dynamic step: 10% of crop size (minimum 10 px)
        step_x = max(10, crop_w // 10)
        step_y = max(10, crop_h // 10)

        self.pan_x += dx_unit * step_x
        self.pan_y += dy_unit * step_y

        # clamp inside bounds
        self.pan_x = max(0, min(self.pan_x, max(0, self.orig_w - crop_w)))
        self.pan_y = max(0, min(self.pan_y, max(0, self.orig_h - crop_h)))

    def reset_pan(self):
        """Center pan to show middle of the frame."""
        if self.zoom_factor <= 1.0:
            self.pan_x = 0
            self.pan_y = 0
            return
        crop_w = max(1, int(self.orig_w / self.zoom_factor))
        crop_h = max(1, int(self.orig_h / self.zoom_factor))
        self.pan_x = (self.orig_w - crop_w) // 2
        self.pan_y = (self.orig_h - crop_h) // 2

    def _on_blur_ksize_change(self, val):
        """Slider callback (tk Scale gives string values). Ensure oddness and clamp."""
        try:
            ks = int(float(val))
        except Exception:
            return
        ks = ensure_odd(ks)
        ks = max(3, min(101, ks))
        self.blur_ksize = ks
        # update slider if we altered it to enforce oddness
        if self.blk_slider.get() != ks:
            self.blk_slider.set(ks)

    def _on_blur_sigma_change(self, val):
        try:
            self.blur_sigma = float(val)
        except Exception:
            pass

    def open_properties_window(self):
        """Shows a Toplevel with runtime properties and some live controls."""
        if self.prop_window and tk.Toplevel.winfo_exists(self.prop_window):
            # bring to front
            self.prop_window.lift()
            return

        w = tk.Toplevel(self.root)
        w.title("Properties")
        w.geometry("360x300")
        self.prop_window = w

        # info labels
        info_fr = ttk.Frame(w, padding=8)
        info_fr.pack(fill="both", expand=True)

        # Runtime info
        def add_label(row, left, right):
            ttk.Label(info_fr, text=left, anchor="w").grid(row=row, column=0, sticky="w", padx=4, pady=2)
            ttk.Label(info_fr, text=right, anchor="e").grid(row=row, column=1, sticky="e", padx=4, pady=2)

        add_label(0, "Source:", str(self.args.source))
        add_label(1, "Orig resolution:", f"{self.orig_w}x{self.orig_h}")
        add_label(2, "Display:", f"{self.display_w}x{self.display_h}")
        add_label(3, "Device:", str(self.device))
        add_label(4, "GPU Blur:", "Yes" if self.use_cuda_blur else "No")
        add_label(5, "Zoom factor:", f"{self.zoom_factor:.2f}")
        add_label(6, "Recording:", "ON" if self.is_recording else "OFF")

        # Provide quick controls: zoom +/- and toggle detector
        ctrl_sub = ttk.Frame(w, padding=(8, 4))
        ctrl_sub.pack(fill="x")
        ttk.Label(ctrl_sub, text="Quick controls:").pack(anchor="w")
        zfr = ttk.Frame(ctrl_sub)
        zfr.pack(fill="x", pady=4)
        ttk.Button(zfr, text="Zoom in", command=lambda: self.set_zoom(self.zoom_factor * 1.25)).pack(side="left", padx=4)
        ttk.Button(zfr, text="Zoom out", command=lambda: self.set_zoom(max(1.0, self.zoom_factor / 1.25))).pack(side="left", padx=4)
        ttk.Button(zfr, text="Reset zoom/pan", command=self.reset_all_view).pack(side="left", padx=4)

        # Live blur controls mirror sliders (for convenience)
        live_fr = ttk.Frame(w, padding=(8, 4))
        live_fr.pack(fill="x")
        ttk.Label(live_fr, text="Live blur (kernel / sigma):").pack(anchor="w")
        live_val = ttk.Label(live_fr, text=f"{self.blur_ksize} / {self.blur_sigma:.1f}")
        live_val.pack(anchor="w", pady=(2, 4))

        # Update the live label periodically for feedback
        def _update_live_label():
            if not (self.prop_window and tk.Toplevel.winfo_exists(self.prop_window)):
                return
            live_val.config(text=f"{self.blur_ksize} / {self.blur_sigma:.1f}")
            self.prop_window.after(300, _update_live_label)
        _update_live_label()

    def set_zoom(self, new_zoom: float):
        """Set zoom factor and clamp reasonable values, center pan."""
        self.zoom_factor = max(1.0, min(new_zoom, 6.0))  # allow up to 6x zoom
        self.reset_pan()

    def reset_all_view(self):
        self.set_zoom(1.0)
        self.reset_pan()

    # ---------------- recording & snapshot ----------------
    def toggle_recording(self):
        if not self.is_recording:
            # start recording to a timestamped file
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(self.args.save_dir, f"output_{ts}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            try:
                self.writer = cv2.VideoWriter(out_path, fourcc, self.fps, (self.orig_w, self.orig_h))
                self.is_recording = True
                print(f"[INFO] Recording started: {out_path}")
            except Exception as e:
                messagebox.showerror("Recording Error", f"Unable to start recording: {e}")
                self.writer = None
                self.is_recording = False
        else:
            # stop
            self.is_recording = False
            if self.writer:
                self.writer.release()
                self.writer = None
            print("[INFO] Recording stopped")

    def take_snapshot(self):
        """Save a snapshot of the currently displayed processed frame (full resolution)."""
        # read a fresh frame for highest fidelity (instead of trying to capture last displayed)
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showwarning("Snapshot", "Unable to capture snapshot (no frame).")
            return
        # apply face blur once more using existing parameters (a simple way)
        processed = frame.copy()
        try:
            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            boxes, _ = self.mtcnn.detect(rgb) if self.mtcnn else (None, None)
        except Exception:
            boxes = None

        if boxes is not None:
            for (x1, y1, x2, y2) in boxes.astype(int):
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(self.orig_w, x2), min(self.orig_h, y2)
                roi = processed[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                try:
                    if self.use_cuda_blur:
                        processed[y1:y2, x1:x2] = blur_roi_torch(roi, self.blur_ksize, self.blur_sigma, self.device)
                    else:
                        processed[y1:y2, x1:x2] = blur_roi_cv2(roi, self.blur_ksize, self.blur_sigma)
                except Exception:
                    processed[y1:y2, x1:x2] = blur_roi_cv2(roi, self.blur_ksize, self.blur_sigma)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(self.args.save_dir, f"snap_{ts}.jpg")
        cv2.imwrite(img_path, processed)
        print(f"[INFO] Snapshot saved {img_path}")
        messagebox.showinfo("Snapshot", f"Saved: {img_path}")

    # ---------------- housekeeping ----------------
    def _on_close(self):
        """Cleanup resources and exit the Tk loop."""
        if not self.running:
            return
        self.running = False

        # close virtual cam
        try:
            if self.cam:
                self.cam.close()
        except Exception:
            pass

        # release writer and cap
        try:
            if self.writer:
                self.writer.release()
                self.writer = None
        except Exception:
            pass

        try:
            if self.cap:
                self.cap.release()
                self.cap = None
        except Exception:
            pass

        # destroy windows / root
        try:
            if self.root:
                # cancel scheduled after if exists
                if hasattr(self, "_after_id") and self._after_id:
                    try:
                        self.root.after_cancel(self._after_id)
                    except Exception:
                        pass
                self.root.destroy()
        except Exception:
            pass

    def _status_text(self):
        return (
            f"Source: {self.args.source}\n"
            f"Device: {self.device}\n"
            f"Resolution: {self.orig_w}x{self.orig_h}\n"
            f"Display: {self.display_w}x{self.display_h}\n"
            f"Blur ks / sigma: {self.blur_ksize} / {self.blur_sigma}\n"
            f"Zoom: {self.zoom_factor:.2f}  Pan: ({self.pan_x},{self.pan_y})\n"
            f"Recording: {'YES' if self.is_recording else 'NO'}"
        )


# ---------- Entrypoint ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Video source (0=webcam, or path to video)")
    parser.add_argument("--save-dir", default="captures", help="Save dir for video/snapshots")
    parser.add_argument("--blur-ksize", type=int, default=41, help="Initial blur kernel size (odd)")
    parser.add_argument("--blur-sigma", type=float, default=12.0, help="Initial blur sigma")
    parser.add_argument("--use-virtualcam", action="store_true", help="Output to virtual webcam (needs pyvirtualcam)")
    args = parser.parse_args()

    # run app
    app = FaceBlurApp(args)
    try:
        app.run_ui()
    except Exception:
        print("[ERROR] App crashed:", traceback.format_exc())
        app._on_close()
        raise


if __name__ == "__main__":
    main()
