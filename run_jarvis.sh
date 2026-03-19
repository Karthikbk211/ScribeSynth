#!/bin/bash

echo "=========================================="
echo "ScribeSynth JarvisLabs H200 Training Setup"
echo "=========================================="

# 1. Activate Environment
# JarvisLabs instances usually come with a good base environment, 
# but if you need to create your specific conda env, do it once:
# conda env create -f environment.yml
eval "$(conda shell.bash hook)"
conda activate scribesynth || echo "Please create the env first if it doesn't exist!"

# 2. Start a Tmux Session (Crucial!)
# If you run a 6-hour job in the Jupyter terminal and close your browser, the job DIES.
# Run this entire script inside a tmux session:
#   $ tmux new -s train
#   $ bash run_jarvis.sh
# To detach and leave it running in the background: Press Ctrl+b, then d.
# To re-attach later: tmux attach -t train

# 3. Run Stage 1 Training
# Notice the use of "\" for multiline commands on Linux instead of Windows' "^"
# torchrun is the preferred launcher on Linux for PyTorch DDP
echo "Starting Stage 1 Training..."
torchrun --nproc_per_node=1 train.py \
    --cfg configs/IAM_scratch.yml \
    --device cuda

# ----------------------------------------------------------------------
# If you ever get disconnected and your Jarvis balance runs out, 
# you can just run this exact script again. The auto-resume logic 
# we added will automatically find the latest checkpoint and continue!
# ----------------------------------------------------------------------

# 4. Fine-Tuning (Stage 2)
# Once Stage 1 is done, you can comment out Stage 1 and run Stage 2:
#
# echo "Starting Stage 2 Fine-Tuning..."
# torchrun --nproc_per_node=1 train_finetune.py \
#     --cfg configs/IAM_finetune.yml \
#     --pretrained output/checkpoints/scribesynth_step500000.pth \
#     --ocr_model model_zoo/text_recognizer.pth \
#     --device cuda
