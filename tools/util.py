import torch
import random
import numpy as np
import os


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_specific_dict(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_dict = model.state_dict()
    filtered = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    return model


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f'checkpoint saved to {path}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
