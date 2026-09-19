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

SENTINEL is a self-hosted, edge-AI surveillance system built for **NVIDIA Jetson** hardware. It fuses classical computer vision with a modern vision-language model to give you a complete picture of your space — who is there, what they are doing, and whether anything needs your attention — all in real time, with **zero cloud dependency** for inference.

Read the full build story: [Why cudaMalloc fails on Jetson Orin Nano Super](https://dev.to/hemkesh2021dotcom/why-cudamalloc-fails-on-nvidia-jetson-orin-nano-super-and-the-one-flag-that-fixes-it-1b0n)

---

## Features

| Capability | Technology |
|---|---|
| Person detection & tracking | YOLOv8n (TensorRT) + ByteTrack |
| Face recognition | DeepFace · Facenet512 · YuNet detector |
| Face Re-ID across occlusions | Session-embedding cosine similarity |
| Periodic re-verification | Stranger verdicts re-checked every 5 s; known faces stay locked until they leave frame |
| Scene understanding | LFM2-VL 1.6B (llama.cpp, runs locally on GPU) |
| Live web dashboard | Flask · MJPEG stream · AI chat · **HTTP Basic Auth** |
| Instant alerts | Telegram Bot API (photo + caption, per-type cooldowns) |
| Entry / exit counting | Horizontal trip-line counter |
| Fire & smoke detection | LFM2-VL scene analysis flag |
| Restricted-hours mode | Escalates alerts between 22:00 – 06:00 |
| Headless operation | Runs without a desktop GUI (default); `--display` for a preview window |

---

## Why This Is Different — Edge Deployment on Shared Memory

The Jetson Orin Nano Super has **no dedicated VRAM** — CPU and GPU share one 8 GB pool (unified memory). A stock CUDA build of llama.cpp tries to allocate GPU tensors with `cudaMalloc` (dedicated-VRAM semantics) and **fails to load LFM2-VL even with several GB free**:

```
NvMapMemAllocInternalTagged: error 12
cudaMalloc failed: out of memory
```

SENTINEL launches llama.cpp with **`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`**, routing allocations through `cudaMallocManaged`. This is the single change that takes LFM2-VL from *failing to load* to *fully GPU-accelerated* on the shared pool — no Ollama, no CPU fallback, no cloud.

> **Because CPU and GPU share 8 GB, startup order matters:** the launcher goes **headless** (frees ~2.5 GB from the desktop), starts **llama-server first** so it claims GPU memory, and only then starts the engine (TensorFlow/PyTorch) and dashboard. `start_sentinel.sh` encodes this order.

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
   │  ByteTrack  │  │  (llama.cpp)   │  │ /tmp/surv_frame │
   └──────┬──────┘  └────────┬───────┘  │  (atomic write) │
          │                  │          └─────────────────┘
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

> **Three processes:** `llama-server` (VLM, C++), `surveillance4_1.py` (engine, Python), `dashboard.py` (Flask). They communicate over localhost HTTP (`:8080`) and two `/tmp` files, written atomically so the dashboard never reads a torn frame.

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

> Works on standard x86 Linux too — for a non-Jetson box, use the YOLO `.pt` model (not the TensorRT `.engine`) and set `device='cpu'` in `surveillance4_1.py`, and run llama.cpp on CPU or a discrete GPU.

### Compatible Cameras

Any camera that supports **ONVIF** or **RTSP** streaming will work. Tested with:

| Camera | Protocol | Resolution | Notes |
|---|---|---|---|
| **Imou Ranger S2** | ONVIF · RTSP | 1080p | Pan/tilt, used in this project |
| Dahua IPC Series | ONVIF · RTSP | Up to 4K | Reliable H.264/H.265 |
| Hikvision DS-2CD Series | ONVIF · RTSP | Up to 4K | Industry standard |
| Reolink RLC Series | RTSP | 1080p – 4K | Budget-friendly |
| Any ONVIF camera | ONVIF · RTSP | 720p+ | Use `subtype=1` for the sub-stream |

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
> On Jetson, OpenCV / TensorFlow / PyTorch come from **JetPack / NVIDIA wheels** — do **not** pip-install them (the PyPI ARM wheels are CPU-only and shadow the CUDA builds). `requirements.txt` lists only the direct deps.

```bash
pip install -r requirements.txt
```

### 4. Configure your environment
```bash
cp .env.example .env
nano .env          # fill in your values (see Configuration below)
chmod 600 .env
```

### 5. Build your face database
Place reference photos in `known_faces/`, one subfolder per person:
```
known_faces/
  Hemkesh/  photo1.jpg  photo2.jpg
  Yogesh/   photo1.jpg
```
Paths are set at the top of `build_face_db.py` (`KNOWN_FACES_DIR`, `DB_PATH`), then:
```bash
python build_face_db.py        # → face_db.pkl (normalised embeddings)
```

### 6. Run everything (recommended — correct order, headless)
```bash
./start_sentinel.sh
```
This frees the desktop GUI, starts **llama-server first** (so it claims GPU memory), then the engine, then the dashboard — each in its own `tmux` session.

<details>
<summary>…or start each part manually</summary>

```bash
# 1) AI engine (VLM) — must start FIRST
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 ./llama-server \
  -m LFM2-VL-1.6B-Q4_0.gguf --mmproj mmproj-LFM2-VL-1.6B-Q8_0.gguf \
  --host 127.0.0.1 --port 8080 --n-gpu-layers 999 --ctx-size 2048 --parallel 1

# 2) surveillance engine (wait until llama is listening)
python surveillance4_1.py

# 3) web dashboard
python dashboard.py
```
The `--mmproj` file is required — without it the server loads text-only and image requests fail.
</details>

### 7. Open the dashboard
The dashboard binds to `127.0.0.1` by default. For a remote computer, open an encrypted SSH tunnel:

```bash
ssh -N -L 5000:127.0.0.1:5000 your-user@your-device
```

Then visit `http://127.0.0.1:5000` on that computer and log in with `DASH_USER` / `DASH_PASS`. The local browser connection travels through the encrypted tunnel. For phone/tablet or shared access, put an HTTPS reverse proxy in front of the loopback listener; do not expose Basic Auth over plain HTTP. `DASH_HOST` can override the bind address only for a deliberately secured deployment.
> Use **Chrome** for the live video (Safari renders MJPEG unreliably).

---

## Configuration (`.env`)

| Variable | Description | Example |
|---|---|---|
| `RTSP_URL` | Full RTSP stream URL with credentials | `rtsp://admin:pass@192.168.1.100:554/cam/realmonitor?channel=1&subtype=1` |
| `BOT_TOKEN` | Telegram bot token from [@BotFather](https://t.me/BotFather) | `123456:ABC-DEF...` |
| `CHAT_ID` | Your Telegram user/group ID | `1223049319` |
| `DASH_USER` | Dashboard login username | `admin` |
| `DASH_PASS` | Dashboard login password (**required**) | `choose-a-strong-one` |

> `YOLO_MODEL` and `FACE_DB_PATH` are set as constants at the top of `surveillance4_1.py` (not env vars). Secrets are loaded from `.env` via `python-dotenv`; `.env` is git-ignored — **never commit it**.

---

## Dashboard

<div align="center">
  <img src="dashboard_demo.png" alt="SENTINEL live dashboard" width="100%">
  <p><em>The SENTINEL dashboard — live feed with detection box, threat / room / person cards,
  live AI scene analysis, the Neural Assistant chat, and the filterable event log.</em></p>
</div>

```
┌─────────────────────────────────────────┐
│  ◉ SENTINEL              CAM-01  LIVE   │
├─────────────────────────────────────────┤
│          [ Live MJPEG Stream ]          │
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
- **Auth** — every route is behind HTTP Basic Auth

---

## Telegram Alerts

| Event | Priority | Condition |
|---|---|---|
| 🔥 Fire / Smoke | P1 — immediate | LFM2 `fire_smoke` flag |
| ⚠️ High / Medium Threat | P2 | LFM2 `harmful` flag |
| 🚨 Intruder | P2 | Stranger + restricted hours (22:00 – 06:00) |

All alerts include a **photo snapshot** and timestamp, with per-type cooldowns so an incident can't spam you.

---

## Project Structure

```
Sentinel_Surveillance/
├── surveillance4_1.py    # Main engine — capture, detection, recognition, VLM, alerts
├── dashboard.py          # Flask web dashboard + AI chat API (Basic Auth)
├── build_face_db.py      # Face enrollment → face_db.pkl (normalised embeddings)
├── start_sentinel.sh     # Launcher — headless → llama → engine → dashboard (correct order)
├── requirements.txt      # Python dependencies (direct only)
├── .env.example          # Environment variable template
├── .gitignore
├── BUILD.md              # TensorRT export + llama.cpp unified-memory build
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

This project is for personal / educational use. Do not deploy in public spaces without complying with local privacy laws — face embeddings are **biometric data** (see DPDP / GDPR / BIPA).

> Note: YOLOv8 / Ultralytics is **AGPL-3.0**. Any closed-source or commercial deployment requires either open-sourcing the derived work or an Ultralytics commercial license.

---

<div align="center">
Built on NVIDIA Jetson &nbsp;·&nbsp; Powered by YOLOv8, DeepFace & LFM2-VL &nbsp;·&nbsp; Alerts via Telegram
</div>


## Dashboard regression checks

The dashboard can be tested without a camera or inference hardware:

```bash
python -m pip install Flask==3.0.3 requests==2.32.3 python-dotenv==1.0.1
python -m unittest discover -s tests
```

The dashboard rejects oversized/invalid chat messages, admits one interactive AI call at a time, and throttles failed authentication attempts per socket address. Behind a proxy, clients may share that address; configure edge authentication/rate limits for a production deployment. The dashboard semaphore does not coordinate other processes using the model. Hardware throughput, alerts, and recognition still require device testing.
