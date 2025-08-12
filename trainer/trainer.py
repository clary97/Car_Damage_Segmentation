# trainer/trainer.py

import os
import time
import numpy as np
import torch
from loguru import logger
from utils.utils import AverageMeter
from utils.metrics import pixel_accuracy, iou_score, dice_coefficient

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
    
    def _train_epoch(self, loader):
        self.model.train()
        losses = AverageMeter()
        for images, masks in loader:
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), images.size(0))
        return losses.avg

    def _validate(self, loader):
        self.model.eval()
        all_outputs, all_targets = [], []
        with torch.no_grad():
            for images, targets in loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        return torch.cat(all_outputs), torch.cat(all_targets)

    def run(self, train_loader, val_loader, test_loader, output_dir):
        best_loss = float('inf')
        best_epoch = 0
        early_stop_counter = 0
        start_time = time.time()

        for epoch in range(1, self.config['epochs'] + 1):
            train_loss = self._train_epoch(train_loader)
            outputs, targets = self._validate(val_loader)
            val_loss = self.criterion(outputs, targets).item()
            iou = iou_score(outputs, targets)
            dice = dice_coefficient(outputs, targets)
            
            #print(f"Epoch {epoch}/{self.config['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | IoU: {iou:.4f} | Dice: {dice:.4f}")

            log_msg = f"Epoch {epoch}/{self.config['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | IoU: {iou:.4f} | Dice: {dice:.4f}"
            logger.info(log_msg)

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                early_stop_counter = 0
                torch.save(self.model.state_dict(), os.path.join(output_dir, "best_model.pth"))
                #print(f"  >> Best model saved. Val Loss: {best_loss:.4f}")
                logger.info(f"  >> Best model saved. Val Loss: {best_loss:.4f}")
            else:
                early_stop_counter += 1

            self.scheduler.step()
            if early_stop_counter >= self.config['early_stop']:
                #print(f"Early stopping at epoch {epoch}.")
                logger.info(f"Early stopping at epoch {epoch}.")
                break
        
        #print("\n--- Testing with best model ---")
        logger.info("\n--- Testing with best model ---")
        self.model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))
        outputs, targets = self._validate(test_loader)
        test_loss = self.criterion(outputs, targets).item()
        test_pa = pixel_accuracy(outputs, targets)
        test_iou = iou_score(outputs, targets)
        test_dice = dice_coefficient(outputs, targets)
        
        #print(f"Test Results -> Loss: {test_loss:.4f}, PA: {test_pa:.4f}, IoU: {test_iou:.4f}, Dice: {test_dice:.4f}")
        logger.info(f"Test Results -> Loss: {test_loss:.4f}, PA: {test_pa:.4f}, IoU: {test_iou:.4f}, Dice: {test_dice:.4f}")

        total_time_str = f"{time.time() - start_time:.2f}s"
        results = {
            'Val_Loss': best_loss, 'Test_Loss': test_loss, 'PA': test_pa,
            'IoU': test_iou, 'Dice': test_dice, 'Best Epoch': best_epoch,
            'Total Time': total_time_str
        }
        return results