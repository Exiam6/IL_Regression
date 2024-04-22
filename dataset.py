import os
import numpy as np
import h5py
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

        return {'image': image, 'target': torch.tensor(target[[0,10]], dtype=torch.float)}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class H5Dataset(Dataset):
    def __init__(self, input_dir, transform=None):
        """
        Args:
            input_dir (string): Directory containing all the .h5 files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.input_dir = input_dir
        self.transform = transform
        self.file_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.h5')]
        self.file_paths.sort()  
        
        
        self.lengths = []
        self.cumulative_lengths = [0]  
        for path in self.file_paths:
            with h5py.File(path, 'r') as file:
                self.lengths.append(len(file['rgb']))
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(file['rgb']))

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        # Find which file contains the index
        file_idx = next(i for i, total in enumerate(self.cumulative_lengths) if total > idx) - 1
        local_idx = idx - self.cumulative_lengths[file_idx] 
      
        with h5py.File(self.file_paths[file_idx], 'r') as file:
            images = file['rgb'][local_idx]
            targets = file['targets'][local_idx, [0, 10]]  #  'steer' and 'speed'

            if self.transform:
                images = self.transform(images)

            return {
                'image': torch.tensor(images, dtype=torch.float).permute(2, 0, 1),  # Adjust for PyTorch: [C, H, W]
                'target': torch.tensor(targets, dtype=torch.float)
            }
