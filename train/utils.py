# train/utils.py
import os
import random
import numpy as np
import torch
import shutil
import importlib

def control_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def copy_sourcefile(output_dir, src_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    dst = os.path.join(output_dir, src_dir)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src_dir, dst)
    print(f'📁 Source files copied to {dst}')


def str_to_class(class_name: str):
    """
    'DeepLab_V3_Plus_Effi_USE_Trans2' 
    """
    module = importlib.import_module(f"models.{class_name}")
    return getattr(module, class_name)