<div align="center">

```
███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗     
██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║     
███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║     
╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║     
███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗
╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝
```

**AI-Powered Home Surveillance System**

*Real-time person detection · Face recognition · Scene analysis · Instant alerts*

---

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Dashboard-000000?style=for-the-badge&logo=flask&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-TensorRT-FF6F00?style=for-the-badge&logo=nvidia&logoColor=white)
![Telegram](https://img.shields.io/badge/Telegram-Alerts-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)
![Jetson](https://img.shields.io/badge/NVIDIA-Jetson_Orin-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

</div>

---

## What is SENTINEL?

SENTINEL is a self-hosted, edge-AI surveillance system built for **NVIDIA Jetson** hardware. It fuses classical computer vision with modern language-vision models to give you a complete picture of your space — who is there, what they are doing, and whether anything needs your attention — all in real time, with **zero cloud dependency** for inference.

Read the full build story: [Why cudaMalloc fails on Jetson Orin Nano Super](https://dev.to/hemkesh2021dotcom/why-cudamalloc-fails-on-nvidia-jetson-orin-nano-super-and-the-one-flag-that-fixes-it-1b0n)

---

## Features

| Capability | Technology |
|---|---|
| Person detection & tracking | YOLOv8n (TensorRT) + ByteTrack |
| Face recognition | DeepFace · Facenet512 · YuNet detector |
| Face Re-ID across occlusions | Session-embedding cosine similarity |
| Scene understanding | LFM2-VL 1.6B (llama.cpp, runs locally) |
| Live web dashboard | Flask · MJPEG stream · AI chat |
| Instant alerts | Telegram Bot API (photo + caption) |
| Entry / exit counting | Horizontal trip-line counter |
| Fire & smoke detection | LFM2-VL scene analysis flag |
| Restricted-hours mode | Escalates alerts between 22:00 – 06:00 |

---
## Why This Is Different — Edge Deployment on Shared Memory

The Jetson Orin Nano Super has **no dedicated VRAM** — CPU and GPU share one 8 GB
pool (unified memory). A stock CUDA build of llama.cpp tries to allocate GPU tensors
with `cudaMalloc` (dedicated-VRAM semantics) and **fails to load LFM2-VL even with
several GB free**:

```
NvMapMemAllocInternalTagged: error 12
cudaMalloc failed: out of memory
```

SENTINEL builds llama.cpp with `GGML_CUDA_ENABLE_UNIFIED_MEMORY=ON`, routing
allocations through `cudaMallocManaged`. This is the single change that takes
LFM2-VL from *failing to load* to *fully GPU-accelerated* on the shared pool — no
Ollama, no CPU fallback, no cloud.

➡️ Full TensorRT export + llama.cpp unified-memory build steps: **[BUILD.md](BUILD.md)**

---

## System Architecture

<div align="center">
  <img src="system_architecture.jpeg" alt="Smart Surveillance System using NVIDIA Jetson Nano" width="100%">
  <p><em>Smart Surveillance System Architecture and Processing Flow</em></p>
</div>

### Component Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                        RTSP Camera                           │
└────────────────────────────┬─────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   FrameReader   │  (background thread)
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
   ┌──────▼──────┐  ┌────────▼───────┐  ┌──────▼──────────┐
   │  YOLOv8n +  │  │  LFM2-VL 1.6B  │  │  Frame Writer   │
   │  ByteTrack  │  │  (llama.cpp)   │  │  /tmp/frame.jpg │
   └──────┬──────┘  └────────┬───────┘  └─────────────────┘
          │                  │
   ┌──────▼──────┐  ┌────────▼───────┐
   │  DeepFace   │  │ Threat / Fire  │
   │  Facenet512 │  │   Evaluator    │
   └──────┬──────┘  └────────┬───────┘
          │                  │
          └────────┬─────────┘
                   │
      ┌────────────▼────────────┐
      │     Dashboard State     │  /tmp/surv_state.json
      └────────────┬────────────┘
                   │
       ┌───────────┴────────────┐
       │                        │
┌──────▼──────┐        ┌────────▼──────┐
│    Flask    │        │   Telegram    │
│  Dashboard  │        │   Bot Alerts  │
│   :5000     │        │               │
└─────────────┘        └───────────────┘
```

---

## Hardware Requirements

> **Tested on:** NVIDIA Jetson Orin Nano Super — 8 GB RAM · This is the hardware this project was built and validated on.

| Component | Tested Hardware | Minimum |
|---|---|---|
| Board | **NVIDIA Jetson Orin Nano Super** | Jetson Nano 4 GB |
| RAM | **8 GB** | 4 GB |
| Storage | **microSD / NVMe SSD** | 32 GB microSD |
| GPU | **1024-core Ampere (Jetson Orin)** | Any CUDA-capable GPU |
| Camera | **Imou Ranger S2 (ONVIF · RTSP)** | Any ONVIF-compatible IP camera |
| JetPack | **6.x** | 5.x |

> Works on standard x86 Linux too — disable TensorRT and switch `device=0` to `device='cpu'` in `surveillance3_10.py`.

### Compatible Cameras

Any camera that supports **ONVIF** or **RTSP** streaming will work with SENTINEL. Tested with:

| Camera | Protocol | Resolution | Notes |
|---|---|---|---|
| **Imou Ranger S2** | ONVIF · RTSP | 1080p | Pan/tilt, used in this project |
| Dahua IPC Series | ONVIF · RTSP | Up to 4K | Reliable H.264/H.265 stream |
| Hikvision DS-2CD Series | ONVIF · RTSP | Up to 4K | Industry standard |
| Reolink RLC Series | RTSP | 1080p – 4K | Budget-friendly option |
| Any ONVIF camera | ONVIF · RTSP | 720p+ | Use subtype=1 for sub-stream |

**Finding your RTSP URL:**
```
# Imou Ranger S2
rtsp://<user>:<pass>@<camera-ip>:554/cam/realmonitor?channel=1&subtype=1

# Hikvision
rtsp://<user>:<pass>@<camera-ip>:554/Streaming/Channels/101

# Reolink
rtsp://<user>:<pass>@<camera-ip>:554/h264Preview_01_sub
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance.git
cd Sentinel_Surveillance
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

**Jetson** (JetPack OpenCV is pre-installed — do NOT pip-install it):
```bash
pip install -r requirements.txt
```

**Standard Linux / x86:**
```bash
pip install -r requirements.txt opencv-python
```

### 4. Configure your environment

```bash
cp .env.example .env
nano .env          # fill in your values (see Configuration section)
```

### 5. Build your face database

Place reference photos in a folder, one subfolder per person:
```
faces/
  Hemkesh/
    photo1.jpg
    photo2.jpg
  Yogesh/
    photo1.jpg
```

Then run the builder script:
```bash
python build_face_db.py --input faces/ --output face_db.pkl
```

### 6. Start the AI engine (LFM2-VL)

```bash
./llama-server -m lfm2-vl-1.6b-q4.gguf --port 8080 --n-gpu-layers 999
```

### 7. Run SENTINEL

Open two terminals:

```bash
# Terminal 1 — surveillance engine
python surveillance3_10.py

# Terminal 2 — web dashboard
python dashboard.py
```

Open in your browser: `http://<device-ip>:5000` or `http://localhost:5000`

---

## Configuration (`.env`)

| Variable | Description | Example |
|---|---|---|
| `RTSP_URL` | Full RTSP stream URL with credentials | `rtsp://admin:pass@192.168.1.100:554/...` |
| `TELEGRAM_BOT_TOKEN` | Token from [@BotFather](https://t.me/BotFather) | `123456:ABC-DEF...` |
| `TELEGRAM_CHAT_ID` | Your Telegram user/group ID | `5227029589` |
| `YOLO_MODEL` | Path to YOLOv8 `.engine` or `.pt` file | `~/yolov8n.engine` |
| `FACE_DB_PATH` | Path to built face database pickle | `~/face_db.pkl` |
| `INTRUDER_LOG` | Path where event log is written | `~/intruder_log.json` |

---

## Dashboard

```
┌─────────────────────────────────────────┐
│  ◉ SENTINEL              CAM-01  LIVE   │
├─────────────────────────────────────────┤
│                                         │
│          [ Live MJPEG Stream ]          │
│                                         │
├──────────┬──────────┬────────┬──────────┤
│  STATUS  │  THREAT  │  ROOM  │ PERSONS  │
├─────────────────────────────────────────┤
│  Detected Persons:  ● Hemkesh ✓  ● Yogesh│
├─────────────────────────────────────────┤
│  Live AI Analysis:  (LFM2-VL scene desc)│
├─────────────────────────────────────────┤
│  NEURAL ASSISTANT  [chat with camera]   │
├─────────────────────────────────────────┤
│  Event Log  [ ALL │ HIGH │ MED │ INFO ] │
└─────────────────────────────────────────┘
```

- **Live stream** — MJPEG feed with auto-reconnect
- **Person chips** — green = known, red = stranger
- **AI chat** — ask the LFM2-VL model anything about the live frame
- **Event log** — filterable intruder history with timestamps

---

## Telegram Alerts

SENTINEL sends alerts automatically:

| Event | Priority | Condition |
|---|---|---|
| 🔥 Fire / Smoke | P1 — immediate | LFM2 fire_smoke flag |
| ⚠️ High / Medium Threat | P2 | LFM2 harmful flag |
| 🚨 Intruder | P2 | Stranger + restricted hours (22:00 – 06:00) |

All alerts include a **photo snapshot** and timestamp.

---

## Project Structure

```
Sentinel_Surveillance/
├── surveillance3_10.py   # Main engine — detection, recognition, alerts
├── dashboard.py          # Flask web dashboard + AI chat API
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
├── .gitignore
├── SETUP_GUIDE.txt       # Full step-by-step setup guide
└── README.md
```

---

## Authors

<div align="center">

| Name | Role | GitHub |
|---|---|---|
| Hemkesh | Lead Developer | [@hemkesh2021-dotcom](https://github.com/hemkesh2021-dotcom) |
| V S Yogeshvar | Co-Developer | [@Yogeshvar425](https://github.com/Yogeshvar425) |

</div>

---

## License

This project is for personal / educational use. Do not deploy in public spaces without complying with local privacy laws.

---

<div align="center">

Built on NVIDIA Jetson &nbsp;·&nbsp; Powered by YOLOv8, DeepFace & LFM2-VL &nbsp;·&nbsp; Alerts via Telegram

</div>
