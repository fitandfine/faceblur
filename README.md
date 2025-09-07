# Real-Time Face Blur with GPU Acceleration + Virtual Webcam

This project provides a Python application that detects faces in real-time from a webcam or video file and automatically blurs them. It is designed for privacy protection during video calls, live streams, or when recording footage. The app supports GPU acceleration (CUDA via PyTorch) for speed, and gracefully falls back to CPU processing if no GPU is available.

---

## Features
- Real-time face detection using **facenet-pytorch (MTCNN)**
- **GPU-accelerated blur** with PyTorch (CUDA if available)
- **CPU fallback blur** using OpenCV when GPU is not available
- Save snapshots (`.jpg`) and recordings (`.mp4`) with one keystroke
- Adjustable blur intensity during runtime
- On-screen overlay showing controls, blur level, and recording status


---

# Real-Time Face Blur with GPU Acceleration + Virtual Webcam – Documentation

This Python application detects faces in real-time from a webcam or video file and blurs them for privacy. It optionally outputs the blurred video to a virtual webcam. It is designed to leverage **GPU acceleration** via PyTorch when available, and gracefully falls back to CPU processing using OpenCV.  

The code is organized with modular functions for clarity, including Gaussian blur generation, face detection, video handling, and user-interactive controls.

---

## Code Overview

### 1. **Imports and Dependencies**
- `argparse` – Handles command-line arguments for input video source, blur parameters, virtual webcam toggle, and save directory.
- `os` – For directory creation and file handling.
- `datetime` – For timestamping recordings and snapshots.
- `cv2` – OpenCV for video capture, display, frame manipulation, and CPU-based blur.
- `numpy` – For array manipulations.
- `torch` – PyTorch, used for GPU-based Gaussian blur.
- `facenet_pytorch` – Provides MTCNN face detector.
- `pyvirtualcam` – Optional library for virtual webcam output.

All imports have error handling for missing packages, allowing the app to run on systems without GPU or optional virtual cam support.

---

### 2. **Gaussian Blur Functions**

#### `make_gaussian_kernel2d(kernel_size: int, sigma: float, device)`
- **Purpose:** Creates a 2D Gaussian kernel tensor for convolution on GPU.
- **Parameters:**
  - `kernel_size` – size of the square kernel (odd number recommended).
  - `sigma` – standard deviation of the Gaussian distribution.
  - `device` – PyTorch device (`cuda` or `cpu`).
- **Logic:**
  1. Ensures kernel size is odd.
  2. Creates coordinate grids for X and Y axes centered at zero.
  3. Computes Gaussian function values using the formula `exp(-(x^2 + y^2)/(2*sigma^2))`.
  4. Normalizes kernel so sum = 1.
  5. Returns a tensor shaped `(1, 1, kernel_size, kernel_size)` ready for depthwise convolution.

#### `blur_roi_torch(roi, kernel_size, sigma, device)`
- **Purpose:** Applies Gaussian blur to a Region of Interest (ROI) using GPU.
- **Parameters:**
  - `roi` – cropped face area (numpy array HxWxC).
  - `kernel_size` and `sigma` – blur parameters.
  - `device` – PyTorch device.
- **Logic:**
  1. Converts ROI from uint8 `[0-255]` to float tensor `[0-1]`.
  2. Reorders dimensions from HxWxC to 1xCxHxW for PyTorch conv2d.
  3. Creates a depthwise 2D Gaussian kernel (repeats kernel for each channel).
  4. Pads the tensor to preserve output size.
  5. Performs depthwise convolution: `conv2d(tensor, kernel, groups=C)`.
  6. Converts tensor back to uint8 numpy array.
- **Benefit:** Fast blurring using GPU, especially for real-time video.

#### `blur_roi_cv2(roi, kernel_size, sigma)`
- **Purpose:** Fallback blur for CPU using OpenCV GaussianBlur.
- **Logic:** 
  1. Ensures kernel size is odd.
  2. Applies `cv2.GaussianBlur` on the ROI.
- **Note:** Slower than GPU convolution for large frames or multiple faces.

---

### 3. **Main Application (`main`)**
The `main()` function orchestrates the application flow:

#### a) **Argument Parsing**
- Uses `argparse` to allow:
  - `--source` – webcam index or video file.
  - `--save-dir` – directory to save videos and snapshots.
  - `--blur-ksize` and `--blur-sigma` – initial blur strength.
  - `--use-virtualcam` – flag to enable virtual webcam output.

#### b) **Setup**
- Creates `captures/` directory if it doesn't exist.
- Chooses device: GPU (`cuda`) if available; otherwise CPU (`cpu`).
- Initializes MTCNN face detector on chosen device.
- Opens video source via OpenCV (`cv2.VideoCapture`).
- Retrieves video frame size and FPS.
- Prints device and source info for user.

#### c) **State Variables**
- `blur_ksize` / `blur_sigma` – current blur settings.
- `is_recording` / `writer` – recording state and OpenCV `VideoWriter`.
- `use_cuda_blur` – boolean indicating if GPU blur should be used.
- `cam` – virtual camera object if enabled.

#### d) **Video Processing Loop**
1. Reads frame from video source.
2. Converts BGR frame to RGB (required by MTCNN).
3. Detects faces using `mtcnn.detect(rgb)`:
   - Returns bounding boxes `(x1, y1, x2, y2)`.
4. Loops over each detected face:
   - Crops the ROI from frame.
   - Applies blur:
     - GPU via `blur_roi_torch` if available.
     - Otherwise CPU via `blur_roi_cv2`.
   - Replaces original ROI with blurred ROI.
5. Draws overlay instructions:
   - Blur adjustment keys `[B/N]`, recording toggle `[R]`, snapshot `[S]`, quit `[Q]`.
   - Shows current blur and recording status.
6. Handles virtual camera output if enabled:
   - Sends processed frame to `pyvirtualcam`.
7. Handles OpenCV window display if virtual cam not used.
8. Handles user key input:
   - `q` / `ESC` → quit loop.
   - `r` → toggle recording, initializes `VideoWriter` if starting.
   - `s` → save snapshot image with timestamp.
   - `b` → increase blur kernel size.
   - `n` → decrease blur kernel size.

#### e) **Cleanup**
- Releases video capture.
- Releases video writer if recording.
- Closes virtual camera.
- Destroys OpenCV windows.

---

### 4. **User Interaction**
- Allows **real-time control** of blur strength.
- Saves recordings and snapshots with timestamps.
- Virtual webcam output enables blurred feed in other applications like Zoom or Teams.

---

### 5. **Logic Flow**
1. Initialize video input and face detector.
2. Read frame → detect faces → apply blur → display or virtual cam output.
3. Handle user inputs for blur, snapshot, recording, or quitting.
4. Loop until user exits or video ends.
5. Clean up resources.

---

### 6. **Error Handling**
- Gracefully handles:
  - Missing GPU: falls back to CPU blur.
  - Missing optional libraries (PyTorch, MTCNN, pyvirtualcam): warns user but continues.
  - Empty or invalid ROIs during face detection.
- Ensures video writer and virtual camera are properly released on exit.

---

### 7. **Key Functions Summary**
| Function | Purpose |
|----------|---------|
| `make_gaussian_kernel2d` | Create 2D Gaussian kernel for GPU convolution. |
| `blur_roi_torch` | Blur a cropped face area using GPU (fast). |
| `blur_roi_cv2` | Blur a cropped face area using CPU (fallback). |
| `main` | Orchestrates video input, face detection, blurring, recording, virtual cam, and user controls. |

---

