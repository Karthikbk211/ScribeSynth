import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import AutoencoderKL
import logging

from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from dataset.loader import IAMDataset, letters
from engine.trainer import Trainer
from network.generator import ScribeSynthGenerator
from network.diffusion import Diffusion
from network.criterions import SupConLoss
from network.text_recognizer import TextRecognizer


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(opt):
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    fix_seed(cfg.TRAIN.SEED)

    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
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
        model.load_state_dict(torch.load(opt.pretrained, map_location='cpu'))
        print(f'loaded pretrained model from {opt.pretrained}')
    else:
        print('no pretrained model loaded, exiting')
        exit()

    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.SOLVER.BASE_LR)

    criterion = dict(nce=SupConLoss(contrast_mode='all'), recon=nn.MSELoss())
    ctc_loss = nn.CTCLoss()
    diffusion = Diffusion(device=device, noise_offset=opt.noise_offset)

    ocr_model = TextRecognizer(nclasses=len(letters), vae=True)
    if len(opt.ocr_model) > 0:
        ocr_model.load_state_dict(
            torch.load(opt.ocr_model, map_location='cpu'), strict=False)
        print(f'loaded ocr model from {opt.ocr_model}')
    else:
        print('no ocr model loaded, exiting')
        exit()
    ocr_model.requires_grad_(False)
    ocr_model = ocr_model.to(device)

    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(filename=f'{cfg.OUTPUT_DIR}/finetune.log', level=logging.INFO)
    logs = logging.getLogger()

    trainer = Trainer(diffusion, model, vae, criterion, optimizer,
                      train_loader, logs, test_loader, device,
                      ocr_model=ocr_model, ctc_loss=ctc_loss)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stable_dif_path', type=str,
                        default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--cfg', dest='cfg_file',
                        default='configs/IAM_finetune.yml')
    parser.add_argument('--pretrained', default='',
                        help='pretrained ScribeSynth model path')
    parser.add_argument('--ocr_model', dest='ocr_model',
                        default='./model_zoo/text_recognizer.pth')
    parser.add_argument('--noise_offset', default=0, type=float)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--local_rank', type=int, default=0)
    opt = parser.parse_args()
    main(opt)
