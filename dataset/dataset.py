# datasets/dataset.py

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from .mask_utils import create_mask

class ImagesDataset(Dataset):
    """
    일반 이미지 + 마스크 데이터셋 (COCO 방식/폴더 방식 모두 지원)
    """
    def __init__(self, image_list, mask_list=None, transform=None, mask_transform=None):
        self.image_list = image_list
        self.mask_list = mask_list
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx]).convert("RGB")
        if self.mask_list is not None:
            mask = Image.open(self.mask_list[idx])
            if self.mask_transform:
                mask = self.mask_transform(mask)
            mask = np.array(mask)
        else:
            mask = None
        if self.transform:
            image = self.transform(image)
        return image, mask

def create_dataloader(image_list, mask_list, batch_size, shuffle=True, transform=None, mask_transform=None, num_workers=4):
    dataset = ImagesDataset(image_list, mask_list, transform, mask_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
