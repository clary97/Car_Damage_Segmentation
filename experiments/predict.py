# predict.py

import argparse
import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model

def predict(args):
    """모델 가중치를 로드하고 단일 이미지 또는 폴더 내 모든 이미지를 예측하여 결과를 저장합니다."""
    
    # 1. 장치 설정
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. 모델 아키텍처 불러오기 및 가중치 로드
    model = get_model('DeepLab_V3_Plus_Effi_USE_Trans2', in_channels=3, num_classes=1).to(device)
    logger.info(f"Loading model weights from {args.weights}")
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # 3. 입력 경로 확인 및 처리할 이미지 리스트 생성
    input_path = args.input
    image_paths = []
    
    if os.path.isdir(input_path):
        # 입력이 폴더인 경우, 내부의 이미지 파일 목록을 모두 가져옴
        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(input_path, filename))
        # 출력 경로가 폴더가 되도록 설정
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Found {len(image_paths)} images in the input folder.")
        
    elif os.path.isfile(input_path):
        # 입력이 단일 파일인 경우
        image_paths.append(input_path)
        output_dir = os.path.dirname(args.output)
        os.makedirs(output_dir, exist_ok=True)
    else:
        logger.error(f"Input path is not a valid file or directory: {input_path}")
        return

    # 4. 각 이미지에 대해 예측 수행 (tqdm으로 진행 바 표시)
    for img_path in tqdm(image_paths, desc="Processing images"):
        input_image_pil = Image.open(img_path).convert("RGB")
        
        # 학습 때와 동일한 전처리
        resized_image = input_image_pil.resize((256, 256), resample=Image.BILINEAR)
        input_tensor = TF.to_tensor(resized_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)

        # 결과 후처리
        pred_mask = torch.sigmoid(output) > 0.5
        pred_mask_np = pred_mask.squeeze().cpu().numpy().astype(np.uint8) * 255
        pred_mask_pil = Image.fromarray(pred_mask_np, mode='L')
        pred_mask_resized = pred_mask_pil.resize(input_image_pil.size, resample=Image.NEAREST)

        # 시각화용 오버레이 이미지 생성
        overlay_mask = pred_mask_resized.convert("RGBA")
        overlay_pixels = overlay_mask.load()
        width, height = overlay_mask.size
        for y in range(height):
            for x in range(width):
                if overlay_pixels[x, y] == (255, 255, 255, 255): # 흰색 픽셀을 반투명 빨간색으로
                    overlay_pixels[x, y] = (255, 0, 0, 128)
                else: # 검은색 픽셀을 투명하게
                    overlay_pixels[x, y] = (0, 0, 0, 0)
        
        visualized_image = Image.alpha_composite(input_image_pil.convert("RGBA"), overlay_mask)
        visualized_image = visualized_image.convert("RGB")

        # 5. 결과 저장
        # 출력 파일 이름 결정
        if os.path.isdir(input_path):
            base_filename = os.path.basename(img_path)
            name, ext = os.path.splitext(base_filename)
            output_filename = f"{name}_pred.png"
            output_path = os.path.join(output_dir, output_filename)
        else: # 단일 파일 입력의 경우
             output_path = args.output
             
        visualized_image.save(output_path)

    logger.info(f"All predictions saved to '{output_dir}'")


if __name__ == '__main__':
    from loguru import logger
    
    parser = argparse.ArgumentParser(description="Inference for Car Damage Segmentation")
    # --input으로  파일/폴더 모두 받을 수 있음
    parser.add_argument('--input', type=str, required=True, help='Path to the input image or folder.')
    parser.add_argument('--weights', type=str, required=True, help='Path to the trained model weights (.pth file).')
    # --output 인자의 기본값을 폴더로 변경
    parser.add_argument('--output', type=str, default='outputs/predictions/', help='Path to save the output file or folder.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID to use.')
    
    args = parser.parse_args()
    predict(args)
