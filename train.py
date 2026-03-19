import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from dataset.loader import IAMDataset
import torch
from engine.trainer import Trainer
from network.generator import ScribeSynthGenerator
from torch import optim
import torch.nn as nn
from network.diffusion import Diffusion, EMA
from diffusers import AutoencoderKL
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from network.criterions import SupConLoss
import random
import numpy as np


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(opt):
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    fix_seed(cfg.TRAIN.SEED)

    backend = 'gloo' if os.name == 'nt' else 'nccl'
    # Set master addr/port explicitly for Windows compatibility
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29500')
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(opt.device, local_rank)

    train_dataset = IAMDataset(
        cfg.DATA_LOADER.IMAGE_PATH,
        cfg.DATA_LOADER.STYLE_PATH,
        cfg.DATA_LOADER.FREQ_PATH,
        cfg.TRAIN.TYPE)
    print(f'number of training samples: {len(train_dataset)}')

    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.IMS_PER_BATCH,
        drop_last=False,
        collate_fn=train_dataset.collate_fn_,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True,
        sampler=train_sampler)

    test_dataset = IAMDataset(
        cfg.DATA_LOADER.IMAGE_PATH,
        cfg.DATA_LOADER.STYLE_PATH,
        cfg.DATA_LOADER.FREQ_PATH,
        cfg.TEST.TYPE)

    test_sampler = DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        drop_last=False,
        collate_fn=test_dataset.collate_fn_,
        pin_memory=True,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        prefetch_factor=3,
        persistent_workers=True,
        sampler=test_sampler)

    model = ScribeSynthGenerator(
        in_channels=cfg.MODEL.IN_CHANNELS,
        model_channels=cfg.MODEL.EMB_DIM,
        out_channels=cfg.MODEL.OUT_CHANNELS,
        num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=cfg.MODEL.NUM_HEADS,
        context_dim=cfg.MODEL.EMB_DIM).to(device)

    if len(opt.pretrained) > 0:
        model.load_state_dict(torch.load(opt.pretrained, map_location='cpu', weights_only=True))
        print(f'loaded pretrained model from {opt.pretrained}')

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # Advanced H200 Speedup: JIT Compilation
    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile() for faster execution.")
    except Exception as e:
        print(f"Warning: torch.compile() failed (likely Windows/Triton issue), skipping... Error: {e}")

    criterion = dict(nce=SupConLoss(contrast_mode='all'), recon=nn.MSELoss())
    optimizer = optim.AdamW(model.parameters(), lr=cfg.SOLVER.BASE_LR)
    diffusion = Diffusion(device=device, noise_offset=opt.noise_offset)

    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device)

    import logging
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(filename=f'{cfg.OUTPUT_DIR}/train.log', level=logging.INFO)
    logs = logging.getLogger()

    trainer = Trainer(diffusion, model, vae, criterion, optimizer,
                      train_loader, logs, test_loader, device)

    if len(opt.resume) > 0:
        trainer.load_checkpoint(opt.resume)
    else:
        import glob
        import re
        checkpoints = glob.glob(f'{cfg.OUTPUT_DIR}/checkpoints/scribesynth_step*.pth')
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(re.search(r'step(\d+)', x).group(1)) if re.search(r'step(\d+)', x) else -1)
            print(f"Auto-resuming from latest checkpoint found: {latest}")
            trainer.load_checkpoint(latest)

    trainer.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stable_dif_path', type=str,
                        default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--cfg', dest='cfg_file',
                        default='configs/IAM_scratch.yml')
    parser.add_argument('--pretrained', default='', help='pretrained model path')
    parser.add_argument('--resume', default='', help='path to checkpoint to resume training')
    parser.add_argument('--noise_offset', default=0, type=float)

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    opt = parser.parse_args()
    main(opt)
