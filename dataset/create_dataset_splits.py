import json
import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def merge_coco_datasets(source_jsons, output_json_path):
    """
    여러 개의 COCO JSON 파일을 ID를 재매핑하여 안전하게 하나로 병합합니다.
    """
    if output_json_path.exists():
        print(f"'{output_json_path.name}' already exists. Skipping merge process.")
        with open(output_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    print("Merging COCO JSON files...")
    
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    img_id_offset = 0
    ann_id_offset = 0

    for json_path in source_jsons:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not merged_data["categories"]:
            merged_data["categories"] = data["categories"]

        old_to_new_img_id = {}
        for image in data["images"]:
            old_id = image["id"]
            new_id = old_id + img_id_offset
            old_to_new_img_id[old_id] = new_id
            image["id"] = new_id
            merged_data["images"].append(image)

        for ann in data["annotations"]:
            ann["id"] += ann_id_offset
            ann["image_id"] = old_to_new_img_id[ann["image_id"]]
            merged_data["annotations"].append(ann)
            
        img_id_offset += len(data["images"])
        ann_id_offset += len(data["annotations"])

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4)
        
    print(f"Successfully merged data into '{output_json_path.name}'.")
    return merged_data

def main():
    # --- 1. 경로 및 설정 정의 ---
    BASE_DIR = Path('dataset/CarDD_release')
    COCO_DIR = BASE_DIR / 'CarDD_COCO'
    ANNOTATION_DIR = COCO_DIR / 'annotations'
    OUTPUT_DIR = BASE_DIR / 'splits'
    
    # 원본 이미지 폴더 경로
    IMAGE_DIRS = {
        'train2017': COCO_DIR / 'train2017',
        'val2017': COCO_DIR / 'val2017',
        'test2017': COCO_DIR / 'test2017'
    }

    # 병합할 원본 JSON 파일 경로
    SOURCE_JSONS = [
        ANNOTATION_DIR / 'instances_train2017.json',
        ANNOTATION_DIR / 'instances_val2017.json',
        ANNOTATION_DIR / 'instances_test2017.json'
    ]
    
    MERGED_JSON_PATH = ANNOTATION_DIR / 'merged_data.json'

    # 분할 설정
    SPLIT_RATIOS = {'training': 0.6, 'validation': 0.2, 'test': 0.2}
    NUM_SPLITS = 10
    RANDOM_SEED_BASE = 42

    # --- 2. COCO 데이터셋 병합 ---
    merged_data = merge_coco_datasets(SOURCE_JSONS, MERGED_JSON_PATH)

    # --- 3. 이미지와 어노테이션을 함께 묶기 ---
    print("Grouping images with their annotations...")
    image_dict = {image["id"]: image for image in merged_data["images"]}
    grouped_data = {}
    for annotation in merged_data["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in grouped_data:
            grouped_data[image_id] = {
                "image": image_dict.get(image_id),
                "annotations": []
            }
        grouped_data[image_id]["annotations"].append(annotation)
    
    all_image_groups = list(grouped_data.values())

    # --- 4. 10개의 다른 데이터 스플릿 생성 ---
    for i in range(1, NUM_SPLITS + 1):
        split_num_str = str(i).zfill(2)
        print(f"\n{'='*20} Creating Split {split_num_str} {'='*20}")
        
        # 매번 다른 시드로 데이터를 섞음
        random.seed(RANDOM_SEED_BASE + i)
        random.shuffle(all_image_groups)

        # 데이터 분할
        total_size = len(all_image_groups)
        train_end = int(total_size * SPLIT_RATIOS['training'])
        val_end = train_end + int(total_size * SPLIT_RATIOS['validation'])
        
        splits = {
            "training": all_image_groups[:train_end],
            "validation": all_image_groups[train_end:val_end],
            "test": all_image_groups[val_end:]
        }
        
        split_output_dir = OUTPUT_DIR / f"split{split_num_str}"

        for split_type, data_groups in splits.items():
            print(f"  Processing '{split_type}' set ({len(data_groups)} images)...")
            
            # 출력 폴더 생성
            target_dir = split_output_dir / split_type
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # 이미지 복사
            for group in tqdm(data_groups, desc=f"    Copying images for {split_type}"):
                image_info = group["image"]
                file_name = image_info["file_name"]
                
                # 원본 이미지 폴더 찾기
                source_path = None
                for dir_name, dir_path in IMAGE_DIRS.items():
                    if (dir_path / file_name).exists():
                        source_path = dir_path / file_name
                        break
                
                if source_path:
                    shutil.copy(source_path, target_dir / file_name)
                else:
                    print(f"Warning: Could not find source for image '{file_name}'")

            # COCO JSON 파일 생성
            split_images = [g["image"] for g in data_groups]
            split_annotations = [ann for g in data_groups for ann in g["annotations"]]
            
            split_json_data = {
                "categories": merged_data["categories"],
                "images": sorted(split_images, key=lambda x: x["id"]),
                "annotations": sorted(split_annotations, key=lambda x: x["id"])
            }
            
            json_output_path = target_dir / f"{split_type}_data.json"
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(split_json_data, f, indent=4)
    
    print("\n✅ All data splits created successfully!")


if __name__ == "__main__":
    main()