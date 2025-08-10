# utils/utils.py
import os
import random
import shutil
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def copy_source_files(src_dirs, output_dir):
    for src in src_dirs:
        dst = os.path.join(output_dir, "src", os.path.basename(src))
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count