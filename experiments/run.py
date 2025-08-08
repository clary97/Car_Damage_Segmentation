# experiments/run.py

import os
import argparse
import pandas as pd
from datetime import datetime
import torch

from config.loader import load_config
from dataset.coco_loader import load_coco_data, create_dataloader
from utils.utils import control_random_seed, copy_sourcefile, str_to_class
from trainer.trainer import Do_Experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepLabV3+ segmentation experiment")
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device IDs to use (e.g., "0" or "0,1")')
    parser.add_argument('--name', type=str, default=None,
                        help='Optional experiment name (default: timestamp-based)')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Set GPU device IDs
    gpu_ids = list(map(int, args.gpu.split(',')))
    config["device_ids"] = gpu_ids

    # Set output folder name
    if args.name:
        exp_name = args.name
    else:
        now = datetime.now()
        exp_name = now.strftime("%y%m%d_%H%M%S")

    model_name = config["model_name"]
    output_root = f'output_proposedd/{model_name}/output_{exp_name}'
    os.makedirs(output_root, exist_ok=True)

    # Initialize result DataFrame
    metrics = [
        'Experiment Time', 'Train Time', 'Iteration', 'Model Name',
        'Val_Loss', 'Test_Loss', 'PA', 'IoU', 'Dice', 'Recall', 'Precision', 'F1 Score',
        'Total Params', 'Train-Prediction Time', 'Best Epoch', 'Time per Epoch',
        'Loss Function', 'LR', 'Batch size', '#Epochs', 'DIR'
    ]
    df = pd.DataFrame(columns=metrics)

    # Run experiments
    for iteration in config["iterations"]:
        print(f"\n========== Iteration {iteration} ==========")
        seed = iteration
        control_random_seed(seed)

        dataset_dir = f'dataset/splits/split{str(iteration).zfill(2)}'
        coco_data = load_coco_data(dataset_dir)

        train_loader = create_dataloader(
            coco_data, "training", os.path.join(dataset_dir, "training"),
            config["batch_size"], aug=True
        )
        val_loader = create_dataloader(
            coco_data, "validation", os.path.join(dataset_dir, "validation"),
            config["batch_size"], aug=False
        )
        test_loader = create_dataloader(
            coco_data, "test", os.path.join(dataset_dir, "test"),
            config["batch_size"], aug=False
        )

        model_class = str_to_class(model_name)
        model = model_class(
            in_channels=config["in_channels"],
            num_classes=config["num_classes"]
        )

        device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)
        else:
            model = model.to(device)

        output_dir = os.path.join(output_root, f'{model_name}_Iter_{iteration}')
        copy_sourcefile(output_dir)

        df = Do_Experiment(
            seed=seed,
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer_type=config["optimizer"],
            lr=config["lr"],
            num_classes=config["num_classes"],
            epochs=config["epochs"],
            metric_names=metrics,
            df=df,
            device=device,
            output_dir=output_dir
        )

        df.to_csv(os.path.join(output_root, f'result_{exp_name}.csv'), index=False, encoding='utf-8-sig')

    print(f"\n✅ All experiments completed. Results saved to {output_root}")


if __name__ == "__main__":
    main()
