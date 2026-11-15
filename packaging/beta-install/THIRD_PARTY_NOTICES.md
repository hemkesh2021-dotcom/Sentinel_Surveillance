# Third-party software and models

**None of what is listed here is inside this package.** The archive holds Sentinel's own files only: the scripts,
their helpers, the manifests, the guide and the application source under `app/`. Those are licensed under the MIT
License ([`LICENSE`](LICENSE), copyright Hemkesh and V S Yogeshvar).

The package's scripts download or build everything else on your device, from the upstream sources named below, and
each item keeps its upstream licence. This file repeats what each upstream states, as read on 2026-11-15. It grants
no permission, changes no licence and is not legal advice: the upstream licence texts govern, and you accept them
when you download and use the items. If you plan anything beyond personal testing, read them first.

## Read these four before you start

1. **Ultralytics (AGPL-3.0).** The detector uses the Python package `ultralytics` 8.4.25 (with
   `ultralytics-thop` 2.0.18) and the weights file `yolov8n.pt`; the TensorRT engine that `build-engine.sh` makes
   on your device is exported from that file. Ultralytics states two options: the GNU Affero General Public License
   version 3, or its own Enterprise License for use outside the AGPL's terms. Sentinel's own source is MIT-licensed
   and is public; a system that runs it together with Ultralytics includes AGPL-3.0 software, and the AGPL's
   conditions apply to that software and to what you derive from it.
   - Licence: <https://github.com/ultralytics/ultralytics/blob/main/LICENSE>
   - Source of the pinned version: <https://github.com/ultralytics/ultralytics/tree/v8.4.25>
   - Licensing options: <https://www.ultralytics.com/license>
2. **Liquid AI LFM3-VL-2B (LFM Open License v1.0).** The scene and chat model and its projector are downloaded
   from Liquid AI's own repository. The licence allows use, reproduction and distribution under its conditions;
   its section 5 makes commercial use conditional on the user's legal entity staying under an annual revenue of
   10 million US dollars, and its section 4 sets conditions for redistribution. This package does not redistribute
   the files.
   - Licence: <https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF/blob/main/LICENSE>
3. **NVIDIA software (NVIDIA's own licence agreements; proprietary).** CUDA, cuDNN, cuDSS, TensorRT and TensorRT's
   Python binding come with JetPack or from NVIDIA's apt repositories, under NVIDIA's software licence agreements.
   The package contains none of them and downloads none of them: you install them, or extract the binding
   (`QUICKSTART.md`, section 2), yourself.
4. **The face model's weights.** `facenet512_weights.h5` is downloaded from the `deepface_models` repository (MIT).
   DeepFace attributes its Facenet models (128 and 512 dimensions) to `davidsandberg/facenet` (MIT) and says of the
   models it wraps: "License types will be inherited when you intend to utilize those models. Please check the
   license types of those models for production purposes."
   - <https://github.com/serengil/deepface_models/blob/master/LICENSE>
   - <https://github.com/davidsandberg/facenet/blob/master/LICENSE.md>
   - <https://github.com/serengil/deepface#licence>

## Files the scripts download or build (`manifest/assets.json`)

Each download is checked against the size and SHA-256 in the manifest; a file that differs is never used.

| Item | From | Revision | Licence, as upstream states it |
|---|---|---|---|
| `LFM3-VL-2B-QAD-Q4_0.gguf`, `mmproj-LFM3-VL-2B-Q8_0.gguf` (scene and chat model, projector) | <https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF> | `7c3b4a43c9023e80bf234fec5e826a3365c8dda8` | LFM Open License v1.0 (`LICENSE` in that repository) |
| `facenet512_weights.h5` (face embedding weights) | <https://github.com/serengil/deepface_models/releases/tag/v1.0> | `v1.0` | repository: MIT, copyright 2019 Sefik Ilkin Serengil; the Facenet model: see item 4 above |
| `face_detection_yunet_2023mar.onnx` (face detector) | <https://github.com/opencv/opencv_zoo/tree/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet> | `f12e12798e8314f7c074a6656816c048dcc95b7a` | MIT, copyright 2020 Shiqi Yu (`LICENSE` in that folder) |
| `yolov8n.pt` (detector weights) | <https://github.com/ultralytics/assets/releases/tag/v8.4.0> | `v8.4.0` | AGPL-3.0 (Ultralytics); see item 1 above |
| `llama-server` and its libraries, **built on your device** from source | <https://github.com/ggml-org/llama.cpp> | commit `19cce6919992fdffd92e0ac152caaf1e2b8f254e` (tag `b11938`) | MIT, copyright 2023-2026 The ggml authors; the source tree carries the licences of the code it includes (`LICENSE`, `licenses/`) |
| `yolov8n.engine`, **built on your device** | exported from `yolov8n.pt` by Ultralytics 8.4.25 with TensorRT 10.3.0 | | derived from the AGPL-3.0 weights; made with NVIDIA TensorRT |

## Python packages (`manifest/requirements-jetson.txt`, `manifest/requirements-pypi.txt`)

`setup-env.sh` installs exactly these versions, each file checked against its SHA-256. Three come from the Jetson AI
Lab index (<https://pypi.jetson-ai-lab.io/jp6/cu126>): `torch`, `torchvision` and `onnxruntime-gpu`. The others come
from PyPI (<https://pypi.org>). The licence column is what each package's own metadata declares in the environment
built from these pins on the reference device. Several of the large packages (NumPy, SciPy, OpenCV, torch,
TensorFlow, h5py, Pillow, ONNX Runtime and others) include further third-party libraries under their own licences;
every installed package keeps its licence files in `<prefix>/venv/lib/python3.10/site-packages/<name>-<version>.dist-info/`.
Each package's page, with its source, is `https://pypi.org/project/<name>/<version>/`.

| Package | Version | Licence, as the package declares it |
|---|---|---|
| absl-py | 2.4.0 | Apache-2.0 |
| annotated-types | 0.7.0 | MIT License |
| astunparse | 1.6.3 | BSD |
| beautifulsoup4 | 4.14.3 | MIT License |
| blinker | 1.9.0 | MIT License |
| certifi | 2026.2.25 | MPL-2.0 |
| charset-normalizer | 3.4.6 | MIT |
| click | 8.3.3 | BSD-3-Clause |
| colorama | 0.4.6 | BSD License |
| coloredlogs | 15.0.1 | MIT |
| contourpy | 1.3.2 | BSD License |
| cycler | 0.12.1 | BSD License |
| deepface | 0.0.99 | MIT |
| filelock | 3.25.2 | MIT |
| fire | 0.7.1 | Apache-2.0 |
| Flask | 3.1.3 | BSD-3-Clause |
| flask-cors | 6.0.2 | MIT |
| flatbuffers | 25.12.19 | Apache 2.0 |
| fonttools | 4.62.1 | MIT |
| fsspec | 2026.2.0 | BSD-3-Clause |
| gast | 0.7.0 | BSD 3-Clause |
| gdown | 5.2.1 | MIT |
| google-pasta | 0.2.0 | Apache 2.0 |
| grpcio | 1.78.0 | Apache-2.0 |
| gunicorn | 25.1.0 | MIT |
| h5py | 3.14.0 | BSD-3-Clause |
| humanfriendly | 10.0 | MIT |
| idna | 3.11 | BSD-3-Clause |
| itsdangerous | 2.2.0 | BSD License |
| Jinja2 | 3.1.6 | BSD License |
| joblib | 1.5.3 | BSD-3-Clause |
| keras | 3.12.1 | Apache License 2.0 |
| kiwisolver | 1.5.0 | BSD License |
| lap | 0.5.13 | BSD-2-Clause |
| libclang | 18.1.1 | Apache License 2.0 |
| lightdsa | 0.0.3 | MIT License |
| lightecc | 0.0.4 | MIT License |
| lightphe | 0.0.20 | MIT License |
| lz4 | 4.4.5 | BSD License |
| markdown-it-py | 3.0.0 | MIT License |
| MarkupSafe | 3.0.3 | BSD-3-Clause |
| matplotlib | 3.10.8 | Python Software Foundation License (matplotlib's own licence agreement) |
| mdurl | 0.1.2 | MIT License |
| ml_dtypes | 0.5.4 | Apache-2.0 |
| mpmath | 1.3.0 | BSD |
| mtcnn | 1.0.0 | MIT |
| namex | 0.1.0 | Apache License 2.0 (its `LICENSE` file; the metadata names none) |
| networkx | 3.4.2 | BSD License |
| numpy | 2.2.6 | BSD License |
| onnx | 1.20.1 | Apache-2.0 |
| onnxruntime | 1.23.2 | MIT License |
| onnxruntime-gpu | 1.23.0 | MIT License |
| onnxslim | 0.1.90 | MIT |
| opencv-python | 4.13.0.92 | Apache 2.0 (with `LICENSE-3RD-PARTY.txt` for the libraries it includes) |
| opt_einsum | 3.4.0 | MIT |
| optree | 0.19.0 | Apache-2.0 |
| packaging | 26.0 | Apache-2.0 OR BSD-2-Clause |
| pandas | 2.3.3 | BSD License |
| pillow | 12.1.1 | MIT-CMU |
| pip | 26.1.2 | MIT |
| polars | 1.39.3 | MIT License |
| polars-runtime-32 | 1.39.3 | MIT License |
| protobuf | 7.34.1 | 3-Clause BSD License |
| psutil | 7.2.2 | BSD-3-Clause |
| pydantic | 2.12.5 | MIT |
| pydantic_core | 2.41.5 | MIT |
| Pygments | 2.17.2 | BSD-2-Clause |
| pyparsing | 3.3.2 | MIT |
| python-dateutil | 2.9.0.post0 | Dual License (BSD License; Apache Software License) |
| python-dotenv | 1.2.2 | BSD-3-Clause |
| pytz | 2024.1 | MIT |
| PyYAML | 6.0.3 | MIT |
| requests | 2.32.5 | Apache-2.0 |
| retina-face | 0.0.17 | MIT License |
| rich | 13.7.1 | MIT |
| scipy | 1.15.3 | BSD License |
| setuptools | 65.5.0 | MIT License |
| six | 1.17.0 | MIT |
| soupsieve | 2.8.3 | MIT |
| sympy | 1.14.0 | BSD |
| tensorflow | 2.21.0 | Apache 2.0 |
| termcolor | 3.3.0 | MIT |
| tf_keras | 2.21.0 | Apache 2.0 |
| torch | 2.9.1 | BSD-3-Clause |
| torchvision | 0.24.1 | BSD |
| tqdm | 4.67.3 | MPL-2.0 AND MIT |
| typing-inspection | 0.4.2 | MIT |
| typing_extensions | 4.15.0 | PSF-2.0 |
| tzdata | 2025.3 | Apache-2.0 |
| ultralytics | 8.4.25 | AGPL-3.0 |
| ultralytics-thop | 2.0.18 | AGPL-3.0 |
| urllib3 | 2.6.3 | MIT |
| Werkzeug | 3.1.6 | BSD-3-Clause |
| wheel | 0.42.0 | MIT License |
| wrapt | 2.1.2 | BSD-2-Clause |

## Needed on the device, not provided or downloaded by this package

| Item | From | Licence |
|---|---|---|
| JetPack 6 components: L4T, CUDA 12.6, cuDNN 9, TensorRT 10.3.0 and its Python binding (`python3-libnvinfer`), cuDSS | NVIDIA (JetPack, <https://repo.download.nvidia.com/jetson/common>, NVIDIA's CUDA apt repository) | NVIDIA's software licence agreements; the binding's own metadata reads "Proprietary" and carries the TensorRT Software License Agreement |
| Python 3.10, OpenBLAS, `python3-cryptography`, git, cmake, a C++ compiler | your distribution's packages | each package's own licence |

## If something here is wrong or out of date

Upstream projects change their terms and their files. The manifests in `manifest/` are what the scripts use; the
upstream pages above are where the licences are. Report an error in this file the way `README.md` describes.
