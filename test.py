import argparse
import os
import torch
from tqdm import tqdm
import torchvision
import torch.distributed as dist
import random
import numpy as np

from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from dataset.loader import Random_StyleIAMDataset, ContentData, generate_type
from network.generator import ScribeSynthGenerator
from network.diffusion import Diffusion
from diffusers import AutoencoderKL


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

    load_content = ContentData()
    total_process = dist.get_world_size()

    text_corpus = generate_type[opt.generate_type][1]
    with open(text_corpus, 'r') as f:
        texts = f.read().split()

    each_process = len(texts) // total_process
    if len(texts) % total_process != 0:
        each_process += 1

    temp_texts = texts[dist.get_rank() * each_process:(dist.get_rank() + 1) * each_process]

    style_dataset = Random_StyleIAMDataset(
        os.path.join(cfg.DATA_LOADER.STYLE_PATH, generate_type[opt.generate_type][0]),
        os.path.join(cfg.DATA_LOADER.FREQ_PATH, generate_type[opt.generate_type][0]),
        len(temp_texts))

    print(f'this process handles {len(style_dataset)} samples')

    style_loader = torch.utils.data.DataLoader(
        style_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        pin_memory=True)

    target_dir = os.path.join(opt.save_dir, opt.generate_type)
    os.makedirs(target_dir, exist_ok=True)

    diffusion = Diffusion(device=opt.device)

    model = ScribeSynthGenerator(
        in_channels=cfg.MODEL.IN_CHANNELS,
        model_channels=cfg.MODEL.EMB_DIM,
        out_channels=cfg.MODEL.OUT_CHANNELS,
        num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=cfg.MODEL.NUM_HEADS,
        context_dim=cfg.MODEL.EMB_DIM).to(opt.device)

    if len(opt.pretrained) > 0:
        model.load_state_dict(torch.load(opt.pretrained, map_location='cpu', weights_only=True))
        print(f'loaded model from {opt.pretrained}')
    else:
        raise IOError('please provide a valid checkpoint path via --pretrained')
    model.eval()

    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(opt.device)

    loader_iter = iter(style_loader)
    for x_text in tqdm(temp_texts, position=0, desc='generating'):
        data = next(loader_iter)
        style_input = data['style'][0].to(opt.device)
        freq_input = data['freq'][0].to(opt.device)
        wid = data['wid']

        data_loader = []
        if len(style_input) > 224:
            data_loader.append((style_input[:224], freq_input[:224], wid[:224]))
            data_loader.append((style_input[224:], freq_input[224:], wid[224:]))
        else:
            data_loader.append((style_input, freq_input, wid))

        for (style_val, freq_val, wid_val) in data_loader:
            text_ref = load_content.get_content(x_text).to(opt.device)
            text_ref = text_ref.repeat(style_val.shape[0], 1, 1, 1)

            x = torch.randn((text_ref.shape[0], 4,
                             style_val.shape[2] // 8,
                             (text_ref.shape[1] * 32) // 8)).to(opt.device)

            if opt.sample_method == 'ddim':
                sampled_images, confidence = diffusion.ddim_sample(
                    model, vae, style_val.shape[0],
                    x, style_val, freq_val, text_ref,
                    opt.sampling_timesteps, opt.eta)
            elif opt.sample_method == 'ddpm':
                sampled_images, confidence = diffusion.ddpm_sample(
                    model, vae, style_val.shape[0],
                    x, style_val, freq_val, text_ref)
            else:
                raise ValueError(f'unknown sample method: {opt.sample_method}')

            for index in range(len(sampled_images)):
                im = torchvision.transforms.ToPILImage()(sampled_images[index])
                image = im.convert("L")
                out_path = os.path.join(target_dir, wid_val[index][0])
                os.makedirs(out_path, exist_ok=True)
                image.save(os.path.join(out_path, x_text + ".png"))

                conf_score = torch.sigmoid(confidence[index]).item()
                with open(os.path.join(out_path, x_text + "_confidence.txt"), 'w') as f:
                    f.write(f'style confidence score: {conf_score:.4f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM_scratch.yml')
    parser.add_argument('--dir', dest='save_dir', default='Generated')
    parser.add_argument('--pretrained', dest='pretrained', default='', required=True)
    parser.add_argument('--generate_type', dest='generate_type', required=True,
                        help='iv_s, iv_u, oov_s, oov_u')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--stable_dif_path', type=str,
                        default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--sample_method', type=str, default='ddim')
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--local_rank', type=int, default=0)
    opt = parser.parse_args()
    main(opt)
