import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize

class NumpyDataset(Dataset):
    def __init__(self, images_file, targets_file, transform=None):
        self.images = np.load(images_file)
        self.targets = np.load(targets_file)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        targets = self.targets[idx,[10]] #speed

        if self.transform:
            image = self.transform(image)

        return {'image': torch.tensor(image, dtype=torch.float).permute(0, 2, 1),  # Adjust for PyTorch: [C, H, W]
                'targets': torch.tensor(targets, dtype=torch.float)}

transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
