# datasets/coco_loader.py

import os
import json
from PIL import Image

def load_coco_data(image_dir, annotation_path):
    """
    COCO 포맷의 어노테이션 파일을 읽고 (이미지 파일, 마스크) 쌍을 리스트로 반환
    """
    with open(annotation_path, 'r') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}
    ann_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        ann_by_image.setdefault(img_id, []).append(ann)

    data_list = []
    for img_id, img_info in images.items():
        filename = img_info['file_name']
        image_path = os.path.join(image_dir, filename)
        anns = ann_by_image.get(img_id, [])
        data_list.append((image_path, anns))
    return data_list
