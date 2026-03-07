# ScribeSynth

A one-shot sentence-level handwriting generation model trained on the CVL dataset. Given a single handwriting sample from a writer, ScribeSynth generates realistic handwriting in that writer's style for any input text.

---

## Overview

ScribeSynth is inspired by one-shot diffusion based handwriting generation. It extends existing approaches with:

- **Sentence-level generation** — supports full sentences instead of just words
- **Learnable high-frequency filter** — replaces the fixed Laplacian kernel with a trainable CNN that learns what stroke details are most useful for style capture
- **Style confidence score** — outputs a confidence score indicating how well the generated handwriting matches the input style
- **IAM dataset support** — trained on the IAM handwriting dataset

---

## Repository Structure

```
ScribeSynth/
├── configs/               # YAML configuration files
│   ├── CVL_scratch.yml    # config for training from scratch
│   └── CVL_finetune.yml   # config for fine-tuning
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

### 3. Download the CVL dataset

Request access and download from the [CVL Database](https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/).

Organize the data as follows:
```
data/
├── images/
│   ├── train/
│   └── test/
├── style/
│   ├── train/
│   └── test/
├── freq/
│   ├── train/
│   └── test/
├── CVL_train.txt
└── CVL_test.txt
```

### 4. Download pretrained VAE

The model uses the VAE from Stable Diffusion v1.5:
```bash
# this is downloaded automatically when you run train.py
# just make sure you have a stable internet connection
```

---

## Training

### Stage 1 — Train from scratch

```bash
python -m torch.distributed.launch --nproc_per_node=1 train.py \
    --cfg configs/CVL_scratch.yml \
    --device cuda
```

### Stage 2 — Fine-tune with recognition loss

```bash
python -m torch.distributed.launch --nproc_per_node=1 train_finetune.py \
    --cfg configs/CVL_finetune.yml \
    --pretrained checkpoints/scribesynth_step50000.pth \
    --ocr_model model_zoo/text_recognizer.pth \
    --device cuda
```

---

## Generation

```bash
python -m torch.distributed.launch --nproc_per_node=1 test.py \
    --cfg configs/CVL.yml \
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
- 15GB+ GPU VRAM recommended (tested on NVIDIA T4)

---

## Acknowledgements

This project is inspired by the One-DM (One-shot Diffusion Mimicker) paper:

> One-Shot Diffusion Mimicker for Handwritten Text Generation. arXiv:2409.04004

The CVL dataset is provided by TU Wien.
