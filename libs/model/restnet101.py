from torchvision.models.segmentation import deeplabv3_resnet101,  DeepLabV3_ResNet101_Weights
import torch.nn as nn

def build_model(device):
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)

    model.classifier[4] = nn.Conv2d(  # Replace the last layer of the classifier for binary segmentation
        in_channels= 256,
        out_channels= 1,  
        kernel_size= 1
    )

    model.to(device)
    return model