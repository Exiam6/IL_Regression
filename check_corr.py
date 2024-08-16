
from train import get_all_y
import argparse
import configs 
from torch.nn.functional import mse_loss
from dataset import ImageTargetDataset, transform,H5Dataset,NumpyDataset 
from model import RegressionResNet 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import train
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = NumpyDataset('/scratch/zz4330/Carla/Train/images.npy', '/scratch/zz4330/Carla/Train/targets.npy',transform=transform)
train_data_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
y_metrics=get_all_y(train_data_loader, device)
all_y=y_metrics['targets']

col1 = all_y[:, 0]
col2 = all_y[:, 1]

# Calculate the correlation coefficient
correlation_matrix = torch.corrcoef(torch.stack((col1, col2)))
correlation_coefficient = correlation_matrix[0, 1]

print('Correlation Coefficient:', correlation_coefficient)
