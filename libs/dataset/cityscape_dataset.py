import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np

class ToRoadMask():
    def __call__(self, label):
        label = np.array(label)
        road_mask = (label == 0).astype(np.float32)  # The BCEWithLogitsLoss() requires the label to be of type float32
        return torch.tensor(road_mask).unsqueeze(0)
    
class CityscapeDataset(Dataset):
    def __init__(self, img_root, label_root, resize=(512, 1024)):  # The original size: (2048, 1024)

        self.img_root = img_root
        self.label_root = label_root
        self.images = []  # used to store the paths of all images in the dataset

        
        for root, _, files in os.walk(img_root):  # Recursively traverses the directory tree starting from img_root and collects the paths of all .png images
            for file in files:
                if file.endswith(".png"):
                    self.images.append(os.path.join(root, file))
        self.img_transform = transforms.Compose(
            [
                transforms.Resize(resize, interpolation=InterpolationMode.BILINEAR),  # Escape the error of OOM, BILINEAR calculates the values by weighting average
                transforms.ToTensor()  # Convert to tensor, normalize by /255, and permute the dim 
            ]
        )

        self.label_transform = transforms.Compose(
            [
                transforms.Resize(resize, interpolation=InterpolationMode.NEAREST),  # NEAREST is used for ensuring no new values are introduced
                ToRoadMask()
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = img_path.replace("leftImg8bit", "gtFine").replace("_gtFine", "_gtFine_labelTrainIds")

        image = Image.open(img_path).convert("RGB")  # Convert to RGB if the image is in a different mode
        label = Image.open(label_path)

        image = self.img_transform(image)
        label = self.label_transform(label)  

        return image, label