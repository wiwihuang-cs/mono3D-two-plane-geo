import torch
from libs.metric.iou import compute_iou
from libs.visualization.loss_visualization import plot_loss

def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    losses = []
    total_iou = 0

    with torch.no_grad():
        for images_batch, labels_batch in dataloader:
            images = images_batch.to(device)
            labels = labels_batch.to(device)

            outputs = model(images)["out"]

            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            
            iou = compute_iou(outputs, labels)
            total_iou += iou
    plot_loss(losses, save_path="results/validation_loss.png")

    avg_iou = total_iou / len(dataloader)
    print(f"Average IoU: {avg_iou:.4f}")
    return avg_iou