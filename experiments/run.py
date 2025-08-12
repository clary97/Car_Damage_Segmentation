# experiments/run.py
import os
import sys
import argparse
import logging
from datetime import datetime
import pandas as pd
import torch
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.loader import load_config
from dataset.dataset import get_dataloaders
from models import get_model
from trainer.trainer import Trainer
from utils.utils import set_seed, copy_source_files
from utils.losses import DiceBCELoss

def main():
    parser = argparse.ArgumentParser(description="Car Damage Segmentation Experiment")
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config file')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    results_df = pd.DataFrame()
    exp_time = datetime.now().strftime("%y%m%d_%H%M%S")
    output_root = f'outputs/output_{exp_time}'
    os.makedirs(output_root, exist_ok=True)

    logger.add(os.path.join(output_root, "experiment.log"))

    start_iter = config["iterations"]["start"]
    end_iter = config["iterations"]["end"]

    for iteration in range(start_iter, end_iter + 1):
        print(f"\n{'='*25} Iteration: {iteration} {'='*25}")
        set_seed(iteration)
        
        # 1. 출력 폴더 설정 및 로거 생성
        iter_output_dir = os.path.join(output_root, f"Iter_{iteration}")
        os.makedirs(iter_output_dir, exist_ok=True)

        # 2. 소스코드 백업
        copy_source_files(['models', 'utils', 'trainer', 'dataset', 'configs'], iter_output_dir)

        # 3. 데이터 로더 준비
        dataset_dir = f'dataset/CarDD_release/splits/split{str(iteration).zfill(2)}'
        train_loader, val_loader, test_loader = get_dataloaders(dataset_dir, config['batch_size'])

        # 4. 모델, 손실함수, 옵티마이저, 스케줄러 준비
        model = get_model(config['model_name'], in_channels=config['in_channels'], num_classes=config['num_classes']).to(device)
        criterion = DiceBCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['scheduler']['T_max'], eta_min=config['scheduler']['eta_min'])

        # 5. Trainer 실행
        trainer = Trainer(model, criterion, optimizer, scheduler, device, config)
        iter_results = trainer.run(train_loader, val_loader, test_loader, iter_output_dir)

        # 6. 최종 결과 기록
        iter_results['Iteration'] = iteration # 결과 딕셔너리에 반복 횟수 추가
        results_df = results_df.append(iter_results, ignore_index=True)
        
        # 매 반복이 끝날 때마다 summary.csv 파일을 덮어쓰기 (실시간 저장 효과)
        summary_csv_path = os.path.join(output_root, 'summary_results.csv')
        results_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        print(f"Iteration {iteration} results saved to {summary_csv_path}")
    
    logger.info(f"\n✅ All iterations complete.")
    #print(f"\n✅ All iterations complete. Results saved in {output_root}")

if __name__ == '__main__':
    main()