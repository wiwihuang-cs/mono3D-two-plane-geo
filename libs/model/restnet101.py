from torchvision.models.segmentation import deeplabv3_resnet101,  DeepLabV3_ResNet101_Weights
import torch.nn as nn
import torch

def build_train_model(device):
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)

    model.classifier[4] = nn.Conv2d(  # Replace the last layer of the classifier for binary segmentation
        in_channels= 256,
        out_channels= 1,  
        kernel_size= 1
    )

    model.to(device)
    return model

def build_inference_model(device, model_path):
    model = deeplabv3_resnet101(weights=None)
    model.classifier[4] = nn.Conv2d(
        in_channels=256,
        out_channels=1,
        kernel_size=1
    )

    checkpoint = torch.load(model_path, map_location=device)  # map_location=device is useed to escape GPU/CPU mismatch issues 
    model.load_state_dict(checkpoint["model"])  # The model is the same as the one used during training, so no need to strict=False

    model.to(device)
    model.eval()

    return model