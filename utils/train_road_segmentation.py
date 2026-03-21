import torch
from libs.dataset.cityscape_dataset import CityscapeDataset
from torch.utils.data import DataLoader
from libs.model.restnet101 import build_model
import torch.nn as nn
import torch.optim as optim
from libs.engine.train import train_one_epoch
from libs.engine.validate import validate_one_epoch

BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 0.001

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }, path)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset
    train_dataset = CityscapeDataset("libs/dataset/cityscape/leftImg8bit/train", "libs/dataset/cityscape/gtFine/train")
    val_dataset = CityscapeDataset("libs/dataset/cityscape/leftImg8bit/val", "libs/dataset/cityscape/gtFine/val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Model
    model = build_model(device)

    # Loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()  # The output of the model is between [-inf, +inf], so we use BCEWithLogitsLoss which combines a sigmoid layer([0, 1]) and the BCELoss.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_iou = 0.0

    # Training loop
    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Training for epoch {epoch+1}/{EPOCHS} completed.")

        val_iou = validate_one_epoch(model, val_loader, loss_fn, device)
        print(f"Validation for epoch {epoch+1}/{EPOCHS} completed.")

        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(model, optimizer, epoch, "models/road_segmentation_best.pth")

if __name__ == "__main__":
    main()