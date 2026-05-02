# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from config import num_epochs, learning_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss Function (BCE + Dice combined)

def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    return 1 - dice.mean()

def bce_dice_loss(pred, target):
    bce  = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


# Metrics

def iou_score(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = (pred + target).clamp(0, 1).sum(dim=(2, 3))
    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()

def pixel_accuracy(pred, target, threshold=0.5):
    pred    = (torch.sigmoid(pred) > threshold).float()
    correct = (pred == target).float().sum()
    return (correct / torch.numel(pred)).item()


# Training Loop

def train_model(model, train_loader, valid_loader, device):
    optimizer    = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses, val_losses = [], []
    train_ious,   val_ious   = [], []

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss, epoch_iou = 0, 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = bce_dice_loss(preds, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_iou  += iou_score(preds, masks)

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_iou  = epoch_iou  / len(train_loader)

        # Validation
        model.eval()
        val_loss, val_iou, val_acc = 0, 0, 0

        with torch.no_grad():
            for imgs, masks in valid_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_loss += bce_dice_loss(preds, masks).item()
                val_iou  += iou_score(preds, masks)
                val_acc  += pixel_accuracy(preds, masks)

        avg_val_loss = val_loss / len(valid_loader)
        avg_val_iou  = val_iou  / len(valid_loader)
        avg_val_acc  = val_acc  / len(valid_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_ious.append(avg_train_iou)
        val_ious.append(avg_val_iou)

        print(f"Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val IoU: {avg_val_iou:.4f} | "
              f"Val Acc: {avg_val_acc:.4f}")

    return train_losses, val_losses, train_ious, val_ious
