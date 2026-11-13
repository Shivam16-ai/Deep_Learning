# VisionPort – Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning

VisionPort is an end-to-end intelligent transportation and computer vision system developed for Windows 10/11 using Python 3.12, Ultralytics YOLOv8, OpenCV, NumPy, ByteTrack, FastAPI, WebSockets, HTML5, CSS3, and Vanilla JavaScript.

---

## 1. Project Directory Structure

```text
VisionPort/
│
├── frontend/                     ← Enterprise Dashboard (HTML5, CSS3, Vanilla JS)
│   ├── index.html                ← Overview Dashboard
│   ├── pages/
│   │   ├── monitoring.html       ← Live video stream with real-time telemetry HUD
│   │   ├── video-analysis.html   ← Video upload / dataset selector & MP4 inference
│   │   ├── detection.html        ← Image detection with before/after comparison
│   │   ├── tracking.html         ← ByteTrack multi-object tracking & trajectory trails
│   │   ├── analytics.html        ← Vehicle counting, density classification & congestion
│   │   ├── alerts.html           ← Active alarms & debounced incident history
│   │   ├── signals.html          ← 3-lamp traffic light widget & state history
│   │   ├── waiting-time.html     ← Stopline queue metrics & average dwell time
│   │   ├── reports.html          ← JSON/CSV/MP4 exports with download links
│   │   └── settings.html         ← Real-time configuration management (YOLO, tracking, ROI)
│   ├── css/
│   │   ├── style.css             ← Design tokens, dark navy & charcoal theme, sidebar/topbar
│   │   ├── components.css        ← Cards, badges, buttons, forms, tables, alerts, toasts
│   │   ├── dashboard.css         ← Responsive grids, video container, signal lamp widget
│   │   └── responsive.css        ← Mobile drawer navigation and responsive breakpoints
│   └── js/
│       ├── api.js                ← REST API client class
│       ├── websocket.js          ← Real-time WebSocket streaming client
│       ├── app.js                ← Global controller, health monitor, navigation
│       ├── dashboard.js          ← Chart.js category distribution & density timeline
│       ├── monitoring.js         ← Live video controls, camera selector, HUD updates
│       ├── detection.js          ← Image dropzone, slider, detections table
│       ├── tracking.js           ← Video tracking playback & trajectory history
│       ├── analytics.js          ← Vehicle tally, density breakdown, congestion charts
│       ├── alerts.js             ← Debounced incident log & alarm severity
│       ├── signals.js            ← 3-lamp traffic light visualization
│       ├── waiting-time.js       ← Queued vehicle dwell time & stopline zone
│       ├── reports.js            ← Generated reports directory & download links
│       └── settings.js           ← Configuration form & live backend sync
│
├── dataset/
│   ├── images/                   ← 20 real traffic photographs (Unsplash/Pexels, CC0)
│   └── videos/                   ← sample_traffic.mp4 (traffic test video)
│
├── captures/                     ← Live webcam captures
│
├── output/
│   ├── detected_images/          ← YOLO-annotated images
│   ├── processed_videos/         ← Analytics-annotated video outputs
│   ├── extracted_frames/         ← Per-frame JPEGs from frame extraction
│   ├── reports/                  ← vehicle_counting, traffic_density, congestion JSON+CSV
│   └── analytics/                ← combined_traffic_analytics JSON+CSV
│
├── models/
│   └── yolov8n.pt                ← 6.25 MB pretrained COCO weights (auto-downloaded)
│
├── modules/
│   ├── __init__.py
│   ├── image_capture.py          ← Task 2: OpenCV webcam capture
│   ├── image_display.py          ← Task 3: Image reading, resizing, display
│   ├── video_processing.py       ← Task 4: Video playback, frame rate control
│   ├── frame_extraction.py       ← Task 5: Video frame extraction
│   ├── vehicle_detection.py      ← Tasks 6-8: YOLOv8 vehicle detection & annotation
│   ├── vehicle_counting.py       ← Task 9: Vehicle counting & category totals
│   ├── traffic_density.py        ← Task 10: Road ROI density classification (LOW/MED/HIGH)
│   ├── congestion_detection.py   ← Task 11: Temporal debouncing & congestion alarms
│   ├── traffic_analytics.py      ← Integrated Tasks 9-11 pipeline orchestrator
│   ├── vehicle_tracking.py       ← Task 12: ByteTrack multi-object tracking & trajectories
│   ├── speed_estimation.py       ← Task 13: Perspective-aware vehicle speed estimation
│   ├── signal_detection.py       ← Task 14: Traffic signal state detection
│   └── waiting_time.py           ← Task 15: Stopline queue waiting time estimation
│
├── utils/
│   ├── __init__.py
│   ├── config.py                 ← Configuration loader & path resolver
│   ├── logger.py                 ← Formatted logging utility
│   └── visualization.py          ← Bounding box, HUD, and trajectory visualizer
│
├── tests/
│   ├── __init__.py
│   ├── test_environment.py       ← Environment & dependency checks
│   ├── test_tasks_1_to_5.py      ← Foundational capture, display & extraction tests
│   ├── test_vehicle_detection.py ← YOLO detection, filtering, and annotation tests
│   ├── test_traffic_analytics.py ← Counting, density, and congestion pipeline tests
│   ├── test_tracking_and_speed.py← ByteTrack IDs and speed estimation tests
│   └── test_server_api.py        ← REST API endpoints & frontend integrity tests
│
├── server.py                     ← FastAPI REST API & WebSocket Server
├── main.py                       ← Unified CLI interface for Tasks 1-13
├── requirements.txt              ← Python dependencies
├── config.yaml                   ← Central project configuration
└── README.md
```

---

## 2. Technical Requirements

- **Operating System:** Windows 10 / Windows 11 (64-bit)
- **Python Version:** 3.12.7 (64-bit)
- **Backend Framework:** FastAPI (`fastapi>=0.115.0`), Uvicorn (`uvicorn[standard]>=0.32.0`)
- **Deep Learning Framework:** Ultralytics YOLO (`ultralytics>=8.3.0`)
- **Computer Vision:** OpenCV (`opencv-python>=4.10.0`)
- **Numerical Processing:** NumPy (`numpy>=1.26.4`)
- **Object Tracking:** ByteTrack with Linear Assignment (`lap>=0.5.14`)
- **Frontend Architecture:** HTML5, CSS3, Vanilla JavaScript (zero React/Next/Vue dependencies)
- **Visualization:** Chart.js (CDN-hosted)
- **Configuration:** YAML (`pyyaml>=6.0.2`)
- **Testing:** Pytest (`pytest>=8.0.0`)

---

## 3. Web Dashboard & REST API Server

VisionPort provides an enterprise web dashboard featuring a dark navy and charcoal UI, real-time telemetry, interactive Chart.js visualizations, and live video streaming.

### Launching the Web Server

From the `VisionPort` directory, start the server:

```powershell
python server.py
```

The server starts locally at:
**`http://127.0.0.1:8000/`**

### Dashboard Pages

| Page | Route | Description |
| :--- | :--- | :--- |
| **Overview** | `/` or `/index.html` | High-level system KPIs, category distribution doughnut chart, density timeline, and recent activity. |
| **Live Monitoring** | `/pages/monitoring.html` | Real-time video player with live HUD stats (FPS, vehicles, density, active alerts) streamed over WebSockets. |
| **Video Analysis** | `/pages/video-analysis.html` | Process recorded MP4 videos with adjustable confidence thresholds and download annotated results. |
| **Vehicle Detection** | `/pages/detection.html` | Image upload or dataset selector with side-by-side comparison and class breakdown table. |
| **Vehicle Tracking** | `/pages/tracking.html` | ByteTrack multi-object tracking visualization, trajectory trails, and unique track records. |
| **Traffic Analytics** | `/pages/analytics.html` | Counting analytics, density classification (LOW / MEDIUM / HIGH), and congestion timelines. |
| **Congestion Alerts** | `/pages/alerts.html` | Temporally debounced congestion alarm logs, severity levels, and incident durations. |
| **Traffic Signals** | `/pages/signals.html` | Interactive 3-lamp traffic light widget (Red / Yellow / Green) and phase transition history. |
| **Waiting Time** | `/pages/waiting-time.html` | Stopline queue dwell time estimation, average and maximum waiting times, and vehicle queue table. |
| **Reports** | `/pages/reports.html` | Export repository listing all generated JSON, CSV, and MP4 files with direct download links. |
| **Settings** | `/pages/settings.html` | Live configuration form for tuning YOLO confidence, IoU, classes, ByteTrack buffers, and speed calibration. |

### REST API Endpoints

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/api/health` | System health check, Python version, YOLO model availability, and device info. |
| `GET` | `/api/dashboard` | Aggregated dashboard telemetry: vehicle counts, density, alerts, and speeds. |
| `GET` | `/api/dataset/images` | List available traffic images in `dataset/images/`. |
| `GET` | `/api/dataset/videos` | List available traffic videos in `dataset/videos/`. |
| `POST` | `/api/detection/image` | Run YOLO detection on an uploaded image or dataset image. Returns detections and annotated image URL. |
| `POST` | `/api/detection/video` | Run YOLO detection and ByteTrack tracking on an MP4 video file. |
| `POST` | `/api/monitoring/start` | Start live video processing session (camera or sample video). |
| `POST` | `/api/monitoring/stop` | Stop active live monitoring session. |
| `GET` | `/api/analytics` | Retrieve vehicle counting, density tiers, and congestion event summaries. |
| `GET` | `/api/vehicles` | Retrieve active ByteTrack tracks and speed estimation records. |
| `GET` | `/api/alerts` | Retrieve active and historical congestion alarms. |
| `GET` | `/api/signals` | Retrieve current traffic signal state (RED / YELLOW / GREEN) and phase history. |
| `GET` | `/api/waiting-time` | Retrieve intersection queue wait times and stationary vehicle records. |
| `GET` | `/api/reports` | List generated output reports with file sizes, timestamps, and download URLs. |
| `GET` | `/api/settings` | Read current configuration from `config.yaml`. |
| `POST` | `/api/settings` | Save updated parameters to `config.yaml`. |
| `WS` | `/ws/monitoring` | WebSocket channel streaming annotated JPEG frames (base64) and telemetry JSON. |

---

## 4. Command Line Interface (CLI)

VisionPort also supports full CLI automation for headless environments:

### Environment & Foundational Operations (Tasks 1–5)

```powershell
# Verify environment and configuration
python main.py --check-env

# Capture a frame from live webcam
python main.py --capture-image

# Read and display a traffic photograph
python main.py --read-image --path "dataset/images/traffic_cars_1.jpg"

# Play a video with frame rate control
python main.py --play-video --path "dataset/videos/sample_traffic.mp4"

# Extract frames from video at configured intervals
python main.py --extract-frames --path "dataset/videos/sample_traffic.mp4"
```

### YOLOv8 Vehicle Detection (Tasks 6–8)

```powershell
# Car detection only (COCO class 2)
python main.py --detect-cars --path "dataset/images/traffic_cars_1.jpg"

# Multi-class detection (car, motorcycle, bus, truck)
python main.py --detect-multi --path "dataset/images/traffic_cars_1.jpg"

# Set confidence threshold and process entire dataset
python main.py --detect-multi --path "dataset/images" --conf 0.50

# Detect vehicles on video stream
python main.py --detect-video --path "dataset/videos/sample_traffic.mp4"
```

### Traffic Analytics (Tasks 9–11)

```powershell
# Vehicle counting per class
python main.py --count-vehicles --path "dataset/images/traffic_cars_1.jpg"

# Traffic density classification (LOW / MEDIUM / HIGH)
python main.py --traffic-density --path "dataset/images/traffic_cars_1.jpg"

# Congestion detection with temporal debouncing
python main.py --detect-congestion --path "dataset/images"

# Full integrated traffic analytics pipeline (Tasks 9, 10 & 11)
python main.py --analyze-traffic --path "dataset/videos/sample_traffic.mp4"
```

### Vehicle Tracking & Speed Estimation (Tasks 12 & 13)

```powershell
# ByteTrack persistent vehicle tracking
python main.py --track-vehicles --path "dataset/videos/sample_traffic.mp4"

# Perspective-aware vehicle speed estimation
python main.py --estimate-speed --path "dataset/videos/sample_traffic.mp4"
```

---

## 5. Automated Test Suite

VisionPort includes comprehensive unit, integration, and API tests covering all modules:

```powershell
python -m pytest -v tests/
```

### Test Coverage Summary:
- `tests/test_environment.py`: Environment validation, paths, and dependencies (5 tests)
- `tests/test_tasks_1_to_5.py`: Camera capture, image display, video playback, frame extraction (5 tests)
- `tests/test_vehicle_detection.py`: YOLO model loading, class filtering, and annotation (6 tests)
- `tests/test_traffic_analytics.py`: Vehicle counting, density classification, and congestion alerts (7 tests)
- `tests/test_tracking_and_speed.py`: ByteTrack ID assignment, trajectories, and speed estimation (24 tests)
- `tests/test_server_api.py`: FastAPI REST API endpoints and frontend file structure integrity (12 tests)

**Total Test Results:** **59 passed in ~33s (100% passing)**

---

## 6. Output Files & Generated Reports

| Output File | Description |
| :--- | :--- |
| `output/detected_images/detected_*.jpg` | Single-class (cars) YOLO annotated photographs |
| `output/detected_images/multi_*.jpg` | Multi-class (car, motorcycle, bus, truck) annotated photographs |
| `output/detected_images/counted_*.jpg` | Annotated images with category count badges |
| `output/detected_images/density_*.jpg` | Annotated images with road ROI and density status |
| `output/detected_images/congestion_*.jpg` | Annotated images with congestion alert banners |
| `output/processed_videos/analytics_*.mp4` | Annotated video output with full counting and density telemetry |
| `output/processed_videos/tracked_traffic.mp4` | Video with persistent ByteTrack tracking IDs and speed badges |
| `output/reports/vehicle_counting_report.json/.csv` | Timestamped per-frame vehicle counts and category totals |
| `output/reports/traffic_density_report.json/.csv` | Density tier history and transition logs |
| `output/reports/congestion_events_report.json/.csv` | Sustained congestion start, end, and duration event logs |
| `output/reports/vehicle_tracking_report.json/.csv` | Persistent vehicle IDs, classes, and trajectory centroid history |
| `output/reports/speed_estimation_report.json/.csv` | Vehicle speed measurements, calibration metadata, and units |
| `output/analytics/combined_traffic_analytics.json/.csv` | Comprehensive multi-modal traffic analytics report |
| `output/extracted_frames/frame_*.jpg` | Sampled raw video frames from extraction |
