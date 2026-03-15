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