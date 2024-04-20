import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageTargetDataset(Dataset):
   
    def __init__(self, images_dir, targets_dir, transform=None):
        """
        Args:
            images_dir (str): Path to the directory containing image files.
            targets_dir (str): Path to the directory containing target files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = images_dir
        self.targets_dir = targets_dir
        self.transform = transform

        self.image_filenames = [f for f in sorted(os.listdir(images_dir)) if f.endswith('.jpeg')]
        self.target_filenames = [f for f in sorted(os.listdir(targets_dir)) if f.endswith('.npy')]

        if len(self.image_filenames) != len(self.target_filenames):
            raise ValueError("The number of images and targets do not match!")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):

        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        target_path = os.path.join(self.targets_dir, self.target_filenames[idx])

        image = Image.open(img_path).convert('RGB')
        target = np.load(target_path)

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'target': torch.tensor(target[0], dtype=torch.float)}

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])