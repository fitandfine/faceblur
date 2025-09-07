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