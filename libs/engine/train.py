from libs.visualization.loss_visualization import plot_loss

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    losses = []

    for images_batch, labels_batch in dataloader:
        images_batch = images_batch.to(device)
        labels_batch = labels_batch.to(device)

        outputs = model(images_batch)["out"]  # Standard output key from the official torchvision segmentation example

        loss = loss_fn(outputs, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())  # .item() to get the scalar value from the tensor
    plot_loss(losses, save_path= "results/training_loss.png")