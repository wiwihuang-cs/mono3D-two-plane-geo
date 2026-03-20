import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from models.road_model import build_model


def load_model(model_path, device):
    model = build_model(num_classes=1)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    model.to(device)
    model.eval()

    return model


def build_transform(resize=(512, 1024)):
    transform = transforms.Compose([
        transforms.Resize(resize, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform


def preprocess_image(image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    original_image = image.copy()

    input_tensor = transform(image).unsqueeze(0).to(device)

    return original_image, input_tensor


def postprocess_output(output, threshold=0.5):
    pred = torch.sigmoid(output)
    pred = (pred > threshold).float()
    pred = pred.squeeze().cpu().numpy()

    return pred


def measure_single_inference_time(model, input_tensor, device):
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        output = model(input_tensor)["out"]

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()

    inference_time = end_time - start_time
    return output, inference_time


def measure_average_inference_time(model, input_tensor, device, warmup_runs=10, measure_runs=50):
    with torch.no_grad():
        # warm-up
        for _ in range(warmup_runs):
            _ = model(input_tensor)["out"]

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        for _ in range(measure_runs):
            _ = model(input_tensor)["out"]

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()

    avg_time = (end_time - start_time) / measure_runs
    fps = 1.0 / avg_time

    return avg_time, fps


def visualize_result(original_image, pred_mask):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # =========================
    # Config
    # =========================
    image_path = "test.png"
    model_path = "best_model.pth"
    resize = (512, 1024)
    threshold = 0.5
    warmup_runs = 10
    measure_runs = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # =========================
    # Load model
    # =========================
    model = load_model(model_path, device)

    # =========================
    # Build transform
    # =========================
    transform = build_transform(resize=resize)

    # =========================
    # Preprocess image
    # =========================
    original_image, input_tensor = preprocess_image(image_path, transform, device)

    print(f"Original image size (W, H): {original_image.size}")
    print(f"Input tensor shape: {input_tensor.shape}")

    # =========================
    # Single inference
    # =========================
    output, single_time = measure_single_inference_time(model, input_tensor, device)
    pred_mask = postprocess_output(output, threshold=threshold)

    print(f"Single inference time: {single_time:.6f} sec")
    print(f"Single inference time: {single_time * 1000:.3f} ms")

    # =========================
    # Average inference benchmark
    # =========================
    avg_time, fps = measure_average_inference_time(
        model=model,
        input_tensor=input_tensor,
        device=device,
        warmup_runs=warmup_runs,
        measure_runs=measure_runs
    )

    print(f"Average inference time over {measure_runs} runs: {avg_time:.6f} sec")
    print(f"Average inference time over {measure_runs} runs: {avg_time * 1000:.3f} ms")
    print(f"FPS: {fps:.2f}")

    # =========================
    # Visualize
    # =========================
    visualize_result(original_image, pred_mask)


if __name__ == "__main__":
    main()