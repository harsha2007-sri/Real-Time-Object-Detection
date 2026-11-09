# Real Object Detection System

A real-time object detection application that captures webcam frames in a browser, sends them to a FastAPI backend, runs YOLO-World plus ByteTrack, and exposes live analytics and historical detection data.

## Features

- Browser webcam access with a responsive dashboard UI
- FastAPI backend with WebSocket streaming
- YOLO-World detection and ByteTrack tracking
- In/out counting with a configurable virtual line
- SQLite-backed detection history and settings persistence
- Dashboard charts and analytics panels
- Mobile-friendly, light-theme interface

## Prerequisites

- Python 3.11 or newer recommended
- Webcam access enabled in the browser
- Optional NVIDIA CUDA GPU support for accelerated inference

## Virtual environment

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the application

```
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open http://localhost:8000 in your browser.

## Model notes

The app loads YOLO-World via Ultralytics and automatically attempts to download `yolov8s-worldv2.pt` from the official model source when it is not already present in the cache. If the download is blocked, manually download the file from the Ultralytics YOLO-World model repository and place it in the cache directory used by the library.

## Notes

- The app uses a configurable detection prompt list tuned for office, mobile, and everyday object categories.
- If CUDA is available, the backend will use the GPU. Otherwise it falls back to CPU.
- The counting line is horizontal and uses upper-to-lower crossing as IN and lower-to-upper crossing as OUT.

## License

This project is provided as a working prototype for demonstration and local development.
