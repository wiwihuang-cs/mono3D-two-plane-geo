import torch

def compute_iou(pred, label, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    # Calculate IoU
    intersection = (pred * label).sum()
    union = pred.sum() + label.sum() - intersection

    return (intersection / (union + 1e-6)).item()  # Add a small epsilon to avoid division by zero