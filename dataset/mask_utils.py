# datasets/mask_utils.py

import numpy as np
from PIL import Image, ImageDraw

def create_mask(image_size, annotations):
    """
    여러 개의 polygon/RLE 어노테이션을 Numpy 배열로 변환.
    """
    mask = Image.new("L", (image_size[0], image_size[1]), 0)
    draw = ImageDraw.Draw(mask)
    for ann in annotations:
        # Polygon 방식
        for poly in ann.get("segmentation", []):
            if isinstance(poly, list):
                draw.polygon(poly, outline=1, fill=1)
    return np.array(mask)