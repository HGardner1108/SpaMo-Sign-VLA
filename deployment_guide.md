# SpaMo ASL-to-English Deployment Guide

Welcome! This guide will walk you through setting up the SpaMo Sign-VLA repository on your local machine to perform **live ASL-to-English translation** using your webcam.

---

## 1. Environment Setup

It is highly recommended to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda to manage your Python environment.

### Clone the Repository
```bash
git clone https://github.com/HGardner1108/SpaMo-Sign-VLA.git
cd SpaMo-Sign-VLA
```

### Create and Activate the Conda Environment
```bash
conda create -n spamo python=3.10 -y
conda activate spamo
```

### Install PyTorch
Install PyTorch with CUDA support. If you are using CUDA 11.8, run:
```bash
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
*(For other CUDA versions, grab the correct command from [pytorch.org](https://pytorch.org/get-started/locally/).)*

### Install Requirements
```bash
pip install -r requirements.txt
pip install opencv-python
```

---

## 2. Download Weights

You will need to download the fine-tuned SpaMo checkpoint to run translations. The underlying LLM (`google/flan-t5-xl`) and feature extractors (`openai/clip-vit-large-patch14`, `MCG-NJU/videomae-large`) will be downloaded automatically by HuggingFace the first time you run the script.

### SpaMo How2Sign Weights
1. Download the custom How2Sign weights from the provided Drive link:
   - **https://drive.google.com/file/d/1huqKMCsJbc0C2zaUiFffqNaYGHKdksJF/view?usp=sharing**
2. Create a `weights` directory in the project root:
   ```bash
   mkdir -p weights
   ```
3. Place the downloaded [.ckpt](file:///home/harry/Documents/SpaMo/SpaMo/weights/spamo.ckpt) file into the `weights` directory and rename it exactly to `spamo_how2sign.ckpt` (e.g. `SpaMo-Sign-VLA/weights/spamo_how2sign.ckpt`).

> **Note:** Make sure you have at least 15-20GB of free disk space for the auto-downloaded HuggingFace models (Flan-T5-XL is ~11GB).

---

## 3. Running the Pipeline

Now you're ready to translate! The pipeline consists of two steps: recording a video and translating it.

### Step A: Record ASL via Webcam
Run the webcam script to capture your signs:
```bash
python Translation_Pipeline/record_webcam.py
```
- A window will pop up showing your webcam feed.
- Perform your ASL sign sequence.
- Press **`q`** on your keyboard to stop recording.
- The video is automatically saved into the `Translation_Pipeline/translation_target/` folder.

### Step B: Translate the Video
Run the inference script to process the newly recorded video:
```bash
python Translation_Pipeline/translate_video.py \
    --video_path Translation_Pipeline/translation_target/ \
    --ckpt_path ./weights/spamo_how2sign.ckpt \
    --config_path ./configs/finetune_how2sign.yaml
```

**What happens under the hood:**
1. The script will crop and extract frames from your video.
2. It automatically extracts spatial (pose/hands) and motion features.
3. The SpaMo adapter fuses these features into the LLM.
4. The console will print the final translated English sentence!

Enjoy your live ASL-to-English translation!
