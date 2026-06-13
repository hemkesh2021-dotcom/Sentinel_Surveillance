# BUILD.md — Engine Export & Inference Setup

This document covers the two build steps the Quick Start glosses over:

1. Exporting YOLOv8n to a **TensorRT `.engine`** for accelerated detection on Jetson.
2. Building **llama.cpp** for LFM2-VL on the Jetson Orin Nano Super — including the
   **unified-memory fix** that is required for the model to load on this hardware.

> **Target hardware:** NVIDIA Jetson Orin Nano Super (8 GB), JetPack 6.x, CUDA arch `sm_87`.

---

## Part 1 — YOLOv8n → TensorRT Engine

The Jetson ships with JetPack's own PyTorch/CUDA stack. Do **not** pip-install a
desktop PyTorch wheel — use the NVIDIA-provided Jetson wheels or the JetPack-bundled
build, otherwise CUDA will not initialise.

### 1.1 Install Ultralytics (Jetson-safe)

```bash
source venv/bin/activate
pip install ultralytics
# OpenCV is pre-installed by JetPack — do NOT pip-install opencv-python on Jetson
```

### 1.2 Export the engine

```bash
yolo export model=yolov8n.pt format=engine half=True device=0 imgsz=640
```

| Flag           | Why                                                            |
| -------------- | ------------------------------------------------------------- |
| `format=engine`| Produces a serialized TensorRT `.engine`                      |
| `half=True`    | FP16 — ~2× throughput on Ampere, negligible accuracy loss     |
| `device=0`     | Build on the Jetson GPU (engines are **not** portable across devices/TensorRT versions) |
| `imgsz=640`    | Must match the inference resolution used in `surveillance3_10.py` |

This writes `yolov8n.engine`. Point `YOLO_MODEL` in your `.env` at it.

> ⚠️ **Engines are not portable.** A `.engine` built on one TensorRT version / GPU
> will fail to load on another. Always rebuild on the deployment device. If you move
> to a new JetPack, re-export.

### 1.3 x86 / no-TensorRT fallback

On a desktop without TensorRT, skip the export and set `YOLO_MODEL=yolov8n.pt`,
then switch `device=0` to `device='cpu'` (or a CUDA index) in `surveillance3_10.py`.

---

## Part 2 — Building llama.cpp for LFM2-VL on Jetson

This is the part that does **not** "just work." On the Orin Nano Super, a stock
CUDA build of llama.cpp fails to load the model even though there is plenty of
free memory. This section documents the failure and the fix.

### 2.1 The problem

The Orin Nano Super uses **unified memory** — the CPU and GPU share one physical
RAM pool; there is no separate VRAM. Stock llama.cpp allocates GPU tensors with
`cudaMalloc` (dedicated-VRAM semantics). On this SoC that allocator path fails
even with several GB free:

```
NvMapMemAllocInternalTagged: 1075072515 error 12
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 661.25 MiB on device 0:
    cudaMalloc failed: out of memory
alloc_tensor_range: failed to allocate CUDA0 buffer of size 693365760
llama_model_load: error loading model: unable to allocate CUDA0 buffer
common_init_from_params: failed to load model
```

`tegrastats` at this point shows the GPU **idle** and RAM mostly **free** —
confirming this is an allocator-semantics issue, not genuine memory exhaustion.

### 2.2 Prerequisites

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git libcurl4-openssl-dev
```

Confirm CUDA is visible:

```bash
nvcc --version          # should report the JetPack CUDA toolkit
```

### 2.3 Clone

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

### 2.4 Build with CUDA + the unified-memory flag

The fix: build with `GGML_CUDA_ENABLE_UNIFIED_MEMORY` so GGML routes GPU
allocations through `cudaMallocManaged` (managed/unified memory) instead of
`cudaMalloc`. On the Orin SoC this lets tensors live in the shared pool.

```bash
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DGGML_CUDA_ENABLE_UNIFIED_MEMORY=ON \
    -DLLAMA_CURL=ON

cmake --build build --config Release -j$(nproc)
```

| Flag                                  | Purpose                                                        |
| ------------------------------------- | -------------------------------------------------------------- |
| `GGML_CUDA=ON`                        | Enable the CUDA backend                                        |
| `CMAKE_CUDA_ARCHITECTURES=87`         | `sm_87` = Orin Ampere. Avoids building dead PTX for other GPUs |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY=ON`  | **The fix** — managed memory instead of dedicated VRAM         |

> The same effect can be forced at runtime without rebuilding by exporting
> `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` before launching the server. Building it in
> is more reliable for an always-on deployment.

### 2.5 Download the model

You need the **base GGUF** plus the **mmproj** (the vision projector — without it
the model is text-only and scene analysis will not work).

```bash
mkdir -p ~/models/lfm2-vl
cd ~/models/lfm2-vl
# from the LiquidAI LFM2-VL GGUF release on Hugging Face:
#   LFM2-VL-1.6B-Q4_0.gguf          (base, ~quantised weights)
#   mmproj-LFM2-VL-1.6B-Q8_0.gguf   (vision projector)
```

### 2.6 Run the server

```bash
~/llama.cpp/build/bin/llama-server \
    --model  ~/models/lfm2-vl/LFM2-VL-1.6B-Q4_0.gguf \
    --mmproj ~/models/lfm2-vl/mmproj-LFM2-VL-1.6B-Q8_0.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --n-gpu-layers 999 \
    --ctx-size 2048
```

`--n-gpu-layers 999` offloads all layers to the GPU. With the unified-memory build
this now succeeds where the stock build failed.

### 2.7 Verify

```bash
curl http://localhost:8080/health        # expect {"status":"ok"}
```

A `200 OK` from `/health` means SENTINEL's scene-analysis worker can reach the model.

---

## Troubleshooting

| Symptom                                              | Cause / Fix                                                                 |
| ---------------------------------------------------- | --------------------------------------------------------------------------- |
| `cudaMalloc failed: out of memory`, GPU idle         | Stock (non-unified) build — rebuild with `GGML_CUDA_ENABLE_UNIFIED_MEMORY=ON`, or export `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` at runtime |
| Engine fails to load after a JetPack/TensorRT update | Re-export the `.engine` on the device — engines are version- and device-specific |
| Scene analysis returns empty / text-only             | `--mmproj` projector not passed or wrong file                               |
| `nvcc: command not found`                            | CUDA toolkit not on `PATH` — `export PATH=/usr/local/cuda/bin:$PATH`        |
| Model loads but is very slow                         | Layers running on CPU — confirm `--n-gpu-layers 999` and a CUDA-enabled build |

---

## Why this matters (the differentiator)

Most "run an LLM on Jetson" guides either use Ollama (which hides this problem) or
silently fall back to CPU. Running LFM2-VL **directly on the GPU via llama.cpp on a
shared-memory Orin SoC** requires the unified-memory allocation path — and that is
the single change that takes the load from *failing* to *fully GPU-accelerated* on
8 GB of shared RAM. That is the core deployment contribution of this project.
