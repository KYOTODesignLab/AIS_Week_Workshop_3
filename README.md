# AIS_Week_Workshop_3

# Working environment
*Miro Board: https://miro.com/app/board/uXjVG1T6fWw=/

# Getting Started

## Software requirements

1. Install the following on your system:


* GIT : https://git-scm.com/

> NOTE: On Mac, GIT is usually pre-installed, you do not need to install it again. To check if it is installed, open a terminal window and type `git`, if the result does not indicate any error, it means you already have it in your system.
>

* Miniconda: https://www.anaconda.com/download/success
>
Make sure to add installation to my PATH variable!
>
<img width="380" height="303" alt="image" src="https://github.com/user-attachments/assets/4cfaa5f0-4e73-40ca-ad57-cb58a58a7b3f" />

* VS Code: https://code.visualstudio.com/

2. Open VS Code, and in the file menu, open your documents folder. Navigate to the terminal window (ctr + @).

If you don't have a GitHub subfolder yet, create it by pasting the following lines:
```
md GitHub
```

Once it is done, navigate to the folder and clone the repository.
```
cd GitHub
git clone https://github.com/grgle/AIS_Week_Workshop_3.git
```

3. In VS Code, set the default terminal profile to command prompt, by clicking on the down arrow next to the + symbol at the top right corner of the terminal window, click "Select Default Profile", and then click "Command Prompt" from the menu in the top middle.

Open your repository folder (`GitHub/AIS_Week_Workshop_3'), and create the necessary virtual environment with conda:

```
conda env create -f environment.yml
```

This environment uses Python 3.11 because the current `ultralytics` and `torch` builds used in this workshop are not reliably available for Python 3.14.

To update the environment with conda, navigate to your repository root folder, and input the following line to the terminal:

```
conda env update -f environment.yml
```

Then activate the environmet:

```
conda activate AIS26
```

If you already created a broken `AIS26` environment from an older `environment.yml`, remove it and recreate it:

```
conda env remove -n AIS26
conda env create -f environment.yml
```

If there is still a problem with `pip` and `ultralytics`, update `pip` inside the environment and install `ultralytics` again:

```
conda activate AIS26
python -m pip install --upgrade pip
python -m pip install ultralytics
```

## Installing PyTorch

PyTorch is **not included** in `environment.yml` because the correct build depends on your operating system, Python version, and whether you have an NVIDIA GPU. Install it manually after creating the conda environment.

### CPU-only (any machine)

Activate the environment and run the command shown by the [official PyTorch selector](https://pytorch.org/get-started/locally/) for your platform with *Package: Pip* and *Compute Platform: CPU*. Example:

```
conda activate AIS26
pip install torch torchvision torchaudio
```

### NVIDIA GPU (CUDA acceleration)

1. **Check your GPU driver** — open a terminal and run:
   ```
   nvidia-smi
   ```
   The top-right corner of the output shows the maximum CUDA version your driver supports (e.g. `CUDA Version: 12.8`).

2. **Install the CUDA Toolkit** that matches your driver:
   https://developer.nvidia.com/cuda-downloads

3. **Activate the environment** and install the matching PyTorch build:
   ```
   conda activate AIS26
   ```
   Open https://pytorch.org/get-started/locally/, select *Package: Pip*, *Language: Python*, *Compute Platform: CUDA XX.X* (matching your driver), and run the generated command. Example for CUDA 12.8:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

4. **Verify the installation:**
   ```
   python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
   ```
   You should see `True` for CUDA if your GPU and driver are set up correctly.

> **Note:** Do not hard-code a specific build like `torch==2.10.0+cu130` in `environment.yml` — it will break for anyone with a different CUDA version or OS.

---

## Repository Structure

```
AIS_Week_Workshop_3/
│
├── 00_Marker_Training/        # YOLO model training pipeline
├── 01_Marker_Recognition/     # Core AR pipeline library
├── 02_Processing_Server/      # Server scripts and web UI
└── docs/                      # GitHub Pages site + image cache
```

---

## `00_Marker_Training/` — Training Pipeline

The folder takes you from raw photos of the markers all the way to a trained `.pt` model file ready to be used by the AR pipeline.

### Step 1 — Collect & resize raw images — `resize.py`

Place raw marker photos (JPEGs) into `labelImg/marker_images/raw/`. Running `resize.py` center-crops each image to a square and rescales it to **1224 × 1224 px**, saving results into `labelImg/marker_images/resized/`. The fixed size matches the training resolution used later, ensuring consistent feature scales.

### Step 2 — Label the resized images — `labelImg/`

Open LabelImg, point it at `resized/`, and draw bounding boxes around each marker in every image. LabelImg saves one `.txt` file per image in YOLO format (class index + normalised box coordinates). Class names are pre-defined in `labelImg/data/predefined_classes.txt`.

The 9 classes are:
`kamogawa · take · zinzya · yama` (the 4 frame markers) and `river · mountain · sunlight · many · big` (geometry + modifier markers).

### Step 3 — Split into train / val sets — `split_dataset.py`

Randomly shuffles the labelled image–label pairs and copies them into:
```
labelImg/dataset/images/train|val
labelImg/dataset/labels/train|val
```
Default split is **80 % train / 20 % val**. Re-run whenever new images are added.

### Step 4 — Train the model — `train_yolo_model.py` + `data.yaml`

Calls `YOLO.train()` with `data.yaml` pointing at the split dataset. Key settings:
- Base weights: `yolov8n.pt` (nano backbone) — or set `START_FROM = output` to resume from the last checkpoint
- 150 epochs, image size 1224, batch 8, GPU if available
- Output saved to `models/AIS26ws3_yolov8n_3.pt` — this is the file loaded by `construct.py`

### Step 5 — Verify the result — `test_with_custom_img.py`

Loads `AIS26ws3_yolov8n_3.pt` and runs inference on 3 randomly sampled images from the resized folder, displaying results on screen. Quick sanity check before deploying.

---

## `01_Marker_Recognition/` — Core AR Pipeline

| File | Purpose |
|---|---|
| `geometry.py` | Defines `CubeSurface` and `Sudare` — the 3-D surface geometry models rendered as AR overlays. Includes an interactive matplotlib viewer when run directly. |
| `interpreter.py` | The full AR engine: `MarkerDetector` (YOLO → bounding boxes), `BaseFrame` (solvePnP pose estimation, improved `fx` via device lookup), `Scene` / `Renderer` (project and draw geometry). |
| `construct.py` | **Entry point for the pipeline.** Loads the YOLO model, defines the marker config, builds the scene, and exposes `process_frame(frame, camera_specs)` — the single function called by all servers. |
| `process_folder.py` | Batch-processes images from `to_process/` → `processed/` using the same pipeline. Run: `python 01_Marker_Recognition/process_folder.py` |
| `elements/models/` | OBJ mesh files used as 3-D overlays. |

---

## `02_Processing_Server/` — Server Modes

| File | Purpose |
|---|---|
| `serve.py` | **Main production server.** Receives phone photos over MQTT (HiveMQ public broker), runs `process_frame`, caches results to `docs/cache/`, posts to Instagram via Cloudinary, and pushes gallery updates via SSE. Run: `python 02_Processing_Server/serve.py` |
| `server_pc_inference.py` | Socket.IO server: phone streams live video, PC runs YOLO inference, streams annotated frames back. Run: `python 02_Processing_Server/server_pc_inference.py` |
| `server_phone_inference.py` | Socket.IO server: phone runs YOLO on-device (ONNX), PC only relays the annotated frames. Run: `python 02_Processing_Server/server_phone_inference.py` |
| `templates/` | HTML/CSS for the phone camera UI (`camera.html`, `camera_pc.html`) and the QR-code index page. |
| `.env.example` | Template for Instagram + Cloudinary credentials — copy to `.env` and fill in values. |

---

## `docs/` — GitHub Pages

| File | Purpose |
|---|---|
| `index.html` | **Phone capture UI** — camera access, rear-lens switching, GPS/orientation collection, device fingerprinting, and chunked MQTT upload with `camera_specs`. |
| `gallery.html` | Public gallery of processed AR images with live SSE updates. |
| `cache/` | Per-session JSON metadata + full-res AR images (written by `serve.py`). |
| `cache_html/` | Downsampled JPEG versions for gallery display. |

