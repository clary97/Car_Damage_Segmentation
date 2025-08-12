# dataset/dataset.py
import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

def _create_mask(image_size, annotations):
    mask = Image.new("L", (image_size[1], image_size[0]), 0)
    draw = ImageDraw.Draw(mask)
    for ann in annotations:
        for polygon in ann["segmentation"]:
            draw.polygon(polygon, outline=1, fill=1)
    return np.array(mask)

class COCODataset(Dataset):
    def __init__(self, images, annotations, image_dir, aug=False, img_size=(256, 256)):
        self.images = list(images.values())
        self.annotations = annotations
        self.image_dir = image_dir
        self.aug = aug
        self.img_height, self.img_width = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info["id"]
        image_path = os.path.join(self.image_dir, image_info["file_name"])

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)

        anns = self.annotations.get(image_id, [])
        mask = _create_mask((image_info["height"], image_info["width"]), anns)
        mask = Image.fromarray(mask).resize((self.img_width, self.img_height), resample=Image.NEAREST)

        if self.aug:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < 0.5:
                angle = random.uniform(-30, 30)
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        image = TF.to_tensor(image).float()
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float()
        mask[mask > 0] = 1.0
        return image, mask

def get_dataloaders(base_dir, batch_size):
    """지정된 디렉토리에서 train, validation, test 데이터 로더를 생성합니다."""
    loaders = {}
    for phase in ["training", "validation", "test"]:
        sub_dir = os.path.join(base_dir, phase)
        json_path = os.path.join(sub_dir, f"{phase}_data.json")

        with open(json_path, 'r') as f:
            coco_data = json.load(f)

        images = {img["id"]: img for img in coco_data["images"]}
        annotations = {}
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            annotations.setdefault(image_id, []).append(ann)
        
        is_train = (phase == "training")
        dataset = COCODataset(images, annotations, image_dir=sub_dir, aug=is_train)
        
        loaders[phase] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=4,
            pin_memory=True
        )

    return loaders['training'], loaders['validation'], loaders['test']