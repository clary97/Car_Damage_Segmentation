# experiments/run.py
import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import torch

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_config
from dataset.dataset import get_dataloaders
from models import get_model
from trainer.trainer import Trainer
from utils.utils import set_seed, copy_source_files
from utils.losses import DiceBCELoss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_config.yaml')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    results_df = pd.DataFrame()
    exp_time = datetime.now().strftime("%y%m%d_%H%M%S")
    output_root = f'outputs/output_{exp_time}'

    for iteration in config["iterations"]:
        print(f"\n{'='*25} Iteration: {iteration} {'='*25}")
        set_seed(iteration)
        
        # 1. 데이터 로더 준비
        dataset_dir = f'CarDD_release/splits/split{str(iteration).zfill(2)}'
        train_loader, val_loader, test_loader = get_dataloaders(dataset_dir, config['batch_size'])
        
        # 2. 모델, 손실함수, 옵티마이저 준비
        model = get_model(config['model_name'], in_channels=config['in_channels'], num_classes=config['num_classes']).to(device)
        
        criterion = DiceBCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['scheduler']['T_max'], eta_min=config['scheduler']['eta_min'])

        # 3. 출력 폴더 설정 및 소스코드 백업
        iter_output_dir = os.path.join(output_root, f"Iter_{iteration}")
        os.makedirs(iter_output_dir, exist_ok=True)
        copy_source_files(['models', 'utils', 'trainer'], iter_output_dir)

        # 4. 학습 실행
        trainer = Trainer(model, criterion, optimizer, scheduler, device, config)
        iter_results = trainer.run(train_loader, val_loader, test_loader, iter_output_dir)
        
        # 5. 결과 기록
        iter_results['Iteration'] = iteration
        results_df = results_df.append(iter_results, ignore_index=True)
        results_df.to_csv(os.path.join(output_root, 'summary.csv'), index=False)
        
    print(f"\n✅ All iterations complete. Results saved in {output_root}")

if __name__ == '__main__':
    main()