import os
import yaml
from easydict import EasyDict as edict

cfg = edict()

# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------
cfg.DATA_LOADER = edict()
cfg.DATA_LOADER.IMAGE_PATH = 'data/images'
cfg.DATA_LOADER.STYLE_PATH = 'data/style'
cfg.DATA_LOADER.FREQ_PATH = 'data/freq'
cfg.DATA_LOADER.NUM_THREADS = 4

# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------
cfg.MODEL = edict()
cfg.MODEL.IN_CHANNELS = 4
cfg.MODEL.OUT_CHANNELS = 4
cfg.MODEL.EMB_DIM = 256
cfg.MODEL.NUM_RES_BLOCKS = 2
cfg.MODEL.NUM_HEADS = 8

# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------
cfg.TRAIN = edict()
cfg.TRAIN.SEED = 42
cfg.TRAIN.IMS_PER_BATCH = 8
cfg.TRAIN.TYPE = 'train'

# -----------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------
cfg.TEST = edict()
cfg.TEST.IMS_PER_BATCH = 4
cfg.TEST.TYPE = 'test'

# -----------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------
cfg.SOLVER = edict()
cfg.SOLVER.BASE_LR = 1e-4

# -----------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------
cfg.OUTPUT_DIR = 'output'


def cfg_from_file(filename):
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.safe_load(f))
    _merge_cfg(yaml_cfg, cfg)


def _merge_cfg(src, dst):
    for key, value in src.items():
        if key not in dst:
            dst[key] = value
        else:
            if isinstance(value, dict):
                _merge_cfg(value, dst[key])
            else:
                dst[key] = value


def assert_and_infer_cfg():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
