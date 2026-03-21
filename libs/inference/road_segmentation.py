from xml.parsers.expat import model

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch
import numpy as np
import cv2

def predict_road(model, image_path, device, resize_size, threshold):
    image = Image.open(image_path).convert("RGB")  # The torch model expects RGB input

    resize = transforms.Resize(
        resize_size,
        interpolation=InterpolationMode.BILINEAR
    )
    resized_image = resize(image)

    transformed_image = transforms.ToTensor()(resized_image)
    input_image = transformed_image.unsqueeze(0).to(device)
    with torch.no_grad(): 
        logits = model(input_image)["out"]
        probs = torch.sigmoid(logits)
        pred_mask = (probs > threshold).float()
    pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)

    return resized_image, pred_mask

def apply_road_mask(resized_image, pred_mask):  
    image = np.array(resized_image)
    road_mask_255 = (pred_mask * 255).astype(np.uint8)
    masked_road = cv2.bitwise_and(image, image, mask=road_mask_255)  # Image and Image when mask is true, else black. The cv2.bitwise_and accepts RGB images.
    return masked_road
