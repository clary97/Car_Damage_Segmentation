# train/trainer.py
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import numpy as np
from monai.metrics import DiceMetric
from monai.metrics import compute_iou
from train.utils import control_random_seed

def DiceBCELoss(pred, target):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    smooth = 1.
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return bce + (1 - dice)

def evaluate(model, dataloader, device):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_list = []
    loss_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)
            loss = DiceBCELoss(outputs, masks)
            loss_list.append(loss.item())

            probs = torch.sigmoid(outputs) > 0.5
            dice_metric(y_pred=probs, y=masks)
            iou = compute_iou(y_pred=probs, y=masks, include_background=False).mean().item()
            iou_list.append(iou)

    dice = dice_metric.aggregate().item()
    iou = np.mean(iou_list)
    val_loss = np.mean(loss_list)
    dice_metric.reset()
    return val_loss, dice, iou

def Do_Experiment(seed, model_name, model, train_loader, val_loader, test_loader,
                  optimizer_type, lr, num_classes, epochs, metric_names, df, device, output_dir):
    print(f"🚀 Training {model_name} with seed {seed}")
    control_random_seed(seed)

    optimizer = getattr(torch.optim, optimizer_type)(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_dice = -1
    best_epoch = 0
    best_model_state = None

    total_start = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}]", leave=False)

        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = DiceBCELoss(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        scheduler.step()

        # Validation
        val_loss, val_dice, val_iou = evaluate(model, val_loader, device)
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            best_model_state = model.state_dict()

        print(f"📌 Epoch {epoch+1}: Val Loss={val_loss:.4f}, Dice={val_dice:.4f}, IoU={val_iou:.4f}")

    total_time = time.time() - total_start

    # evaluate : test dataset
    model.load_state_dict(best_model_state)
    test_loss, test_dice, test_iou = evaluate(model, test_loader, device)

    # save results
    result = {
        'Experiment Time': time.strftime('%y%m%d_%H%M%S'),
        'Train Time': f"{total_time:.1f}s",
        'Iteration': seed,
        'Model Name': model_name,
        'Val_Loss': f"{val_loss:.4f}",
        'Test_Loss': f"{test_loss:.4f}",
        'PA': '-',
        'IoU': f"{test_iou:.4f}",
        'Dice': f"{test_dice:.4f}",
        'Recall': '-', 'Precision': '-', 'F1 Score': '-',
        'Total Params': sum(p.numel() for p in model.parameters()),
        'Train-Prediction Time': '-', 'Best Epoch': best_epoch,
        'Time per Epoch': f"{total_time / epochs:.2f}",
        'Loss Function': 'DiceBCELoss',
        'LR': lr,
        'Batch size': train_loader.batch_size,
        '#Epochs': epochs,
        'DIR': output_dir
    }

    df.loc[len(df)] = result
    return df
