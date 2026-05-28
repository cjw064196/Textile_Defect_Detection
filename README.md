# Textile Defect Detection System

A web-based defect detection system for textile quality control, built with Django and YOLO. Supports real-time image, video, and camera detection with a multi-role user management interface.

---

## Features

- **Image Detection** — Upload fabric images and get annotated results with defect type, location, and confidence score
- **Video Detection** — Process video files frame-by-frame and export annotated result videos
- **Camera Detection** — Real-time defect detection via live camera feed
- **Detection History** — All detection results are saved and searchable per user
- **Knowledge Base** — Admin-managed articles on textile defect types
- **User Management** — Role-based access control (admin / regular user)
- **Model Switching** — Admins can upload and switch custom `.pt` model files at runtime

---

## Detectable Defect Types

| Class | Description |
|---|---|
| `delik` | Holes or tears in the fabric |
| `dokuma_iplik_hata` | Weaving thread defects (broken threads, skip stitches) |
| `leke` | Surface stains or contamination |
| `topbasi` | Exposed yarn ends on the fabric surface |

---

## Tech Stack

- **Backend**: Django 5.x, Python 3.10
- **Detection**: YOLOv8 / YOLO11 / YOLO12 (via Ultralytics)
- **Database**: SQLite
- **Frontend**: Bootstrap 5, HTML/CSS/JS

---

## Requirements

| Item | Requirement |
|---|---|
| OS | macOS / Windows / Linux |
| Python | 3.10 (recommended) |
| Package Manager | Anaconda / Miniconda |
| Hardware | CPU is sufficient; NVIDIA GPU accelerates inference |

---

## Quick Start

### 1. Create and activate a virtual environment

```bash
conda create -n textile python=3.10 -y
conda activate textile
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install ultralytics
```

### 3. Initialize the database

```bash
python manage.py migrate
python manage.py shell < newsite/createadmin.py
```

This creates a default admin account: **username** `admin` / **password** `admin`.

### 4. Run the server

```bash
python manage.py runserver
```

Open your browser at **http://127.0.0.1:8000**

---

## Default Accounts

| Username | Password | Role |
|---|---|---|
| `admin` | `admin` | Administrator — manage users, records, and knowledge base |
| *(register)* | — | Regular user — run detections and view personal history |

---

## Project Structure

```
Textile_Defect_Detection/
├── newsite/            # Django app (views, models, routes, settings)
├── templates/          # HTML page templates
├── static/             # CSS, JS, fonts, and static assets
├── ultralytics/        # Local Ultralytics library (customized)
├── test_media/         # Sample images and video for testing
├── models/             # Trained model weights (best.pt) — not tracked by git
├── pretrained/         # Base YOLO pretrained weights — not tracked by git
├── train_data/         # Training dataset — not tracked by git
├── runs/               # Training output — not tracked by git
├── yolo_detection.py   # YOLO inference core (model loading, drawing, caching)
├── config_text.py      # UI text config and class name mapping (EN/CN)
├── run_train.py        # Standalone model training script
├── run_detect_video.py # Standalone video detection script
├── run_detect_camera.py# Standalone camera detection script
└── requirements.txt
```

---

## Model Files

The following files are excluded from this repository due to their size. Download or place them manually:

| Path | Description |
|---|---|
| `models/best.pt` | **Used by default** — custom model trained on the textile dataset |
| `pretrained/yolov8n.pt` | YOLOv8 base pretrained weights |
| `pretrained/yolo11n.pt` | YOLO11 base pretrained weights |
| `pretrained/yolo12n.pt` | YOLO12 base pretrained weights |

---

## Retraining the Model (Optional)

```bash
conda activate textile
python run_train.py
```

Training output is saved to `runs/`. Copy the best weights to `models/best.pt` to use them in the web app.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'django'` | Run `conda activate textile` first |
| `No module named 'ultralytics'` | Run `pip install ultralytics` |
| Chinese labels show as squares | Ensure `simhei.ttf` exists in the project root |
| First detection is slow (5–15s) | Normal — the model loads on first request and is cached after |
