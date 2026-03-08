# ScribeSynth

A one-shot sentence-level handwriting generation model trained on the IAM Handwriting Database. Given a single handwriting sample from a writer, ScribeSynth generates realistic handwriting in that writer's style for any input text.

---

## Overview

ScribeSynth is inspired by one-shot diffusion based handwriting generation. It extends existing approaches with:

- **Sentence-level generation** — supports full sentences instead of just words
- **Learnable high-frequency filter** — replaces the fixed Laplacian kernel with a trainable CNN that learns what stroke details are most useful for style capture
- **Style confidence score** — outputs a confidence score indicating how well the generated handwriting matches the input style
- **IAM dataset support** — trained on the IAM Handwriting Database

---

## Repository Structure

```
ScribeSynth/
├── configs/               # YAML configuration files
│   ├── IAM_scratch.yml    # config for training from scratch
│   └── IAM_finetune.yml   # config for fine-tuning
├── dataset/               # dataset loading
│   └── loader.py
├── engine/                # training loop
│   └── trainer.py
├── network/               # model architecture
│   ├── attention.py       # transformer building blocks
│   ├── criterions.py      # loss functions
│   ├── diffusion.py       # diffusion process and sampling
│   ├── feature_extractor.py  # ResNet18 with dilated convolutions
│   ├── generator.py       # UNet denoising backbone
│   ├── style_content_mixer.py  # style and content fusion
│   └── text_recognizer.py  # HTR network for legibility loss
├── tools/                 # utility functions
├── parse_config.py        # config parser
├── train.py               # training from scratch
├── train_finetune.py      # fine-tuning with recognition loss
├── test.py                # inference and generation
├── environment.yml        # conda environment
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Karthikbk211/ScribeSynth.git
cd ScribeSynth
```

### 2. Create the environment

```bash
conda env create -f environment.yml
conda activate scribesynth
```

### 3. Download the IAM Handwriting Database

The IAM dataset is available on Kaggle:

> [IAM Handwriting Database — Kaggle](https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database)

After downloading and extracting, organize the data as follows:

```
data/
├── images/
│   ├── train/
│   │   └── <writer_id>/
│   │       └── <image>.png
│   └── test/
│       └── <writer_id>/
│           └── <image>.png
├── style/
│   ├── train/
│   └── test/
├── freq/
│   ├── train/
│   └── test/
├── IAM_train.txt
└── IAM_test.txt
```

Each line in `IAM_train.txt` / `IAM_test.txt` should follow the format:
```
<writer_id>,<image_name> <transcription>
```

### 4. Pretrained VAE

The model uses the VAE from Stable Diffusion v1.5. It is downloaded automatically when you run `train.py` (requires a stable internet connection):

```bash
# downloaded automatically from huggingface.co/runwayml/stable-diffusion-v1-5
```

---

## Training

### Stage 1 — Train from scratch

```bash
torchrun --nproc_per_node=1 train.py \
    --cfg configs/IAM_scratch.yml \
    --device cuda
```

### Resume from checkpoint

```bash
torchrun --nproc_per_node=1 train.py \
    --cfg configs/IAM_scratch.yml \
    --resume checkpoints/scribesynth_step15000.pth \
    --device cuda
```

### Stage 2 — Fine-tune with recognition loss

```bash
torchrun --nproc_per_node=1 train_finetune.py \
    --cfg configs/IAM_finetune.yml \
    --pretrained checkpoints/scribesynth_step50000.pth \
    --ocr_model model_zoo/text_recognizer.pth \
    --device cuda
```

---

## Generation

```bash
torchrun --nproc_per_node=1 test.py \
    --cfg configs/IAM_scratch.yml \
    --pretrained checkpoints/scribesynth_final.pth \
    --generate_type iv_s \
    --sample_method ddim \
    --dir Generated
```

Generation types:
- `iv_s` — in-vocabulary, seen writers
- `iv_u` — in-vocabulary, unseen writers
- `oov_s` — out-of-vocabulary, seen writers
- `oov_u` — out-of-vocabulary, unseen writers

---

## How It Works

1. **Style encoding** — two handwriting samples from the same writer are passed through two parallel ResNet18 encoders. One captures low-frequency style (overall slant, weight, spacing) and the other captures high-frequency details (stroke edges, curves) through a learnable filter.

2. **Content encoding** — the desired text is rendered as unifont bitmap images and encoded through a ResNet18.

3. **Style-content fusion** — a transformer decoder fuses the content with style features in two stages — first low frequency, then high frequency.

4. **Diffusion generation** — the fused style-content context guides a UNet to denoise a random latent into a clean handwriting image over 50 DDIM steps.

5. **VAE decoding** — the clean latent is decoded by a frozen VAE into the final handwriting image.

---

## Requirements

- Python 3.9
- PyTorch 2.0.1
- CUDA 11.7
- 8GB+ GPU VRAM minimum (tested on NVIDIA RTX 4060 8GB)
- Recommended: set `IMS_PER_BATCH: 2` in config for 8GB GPUs

---

## Acknowledgements

This project is inspired by the One-DM (One-shot Diffusion Mimicker) paper:

> One-Shot Diffusion Mimicker for Handwritten Text Generation. arXiv:2409.04004

The IAM Handwriting Database is provided by the Research Group on Computer Vision and Artificial Intelligence, University of Bern.
