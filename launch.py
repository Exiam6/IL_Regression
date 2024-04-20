import argparse
import configs 
from torch.nn.functional import mse_loss
from dataset import ImageTargetDataset, transform  
from model import RegressionResNet 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import train

def set_seed(random_seed):
    torch.manual_seed(random_seed)
def set_optimizer(args,model):
    if args.case2:
        param_groups = [
        {'params': model.model.fc.parameters(), 'weight_decay': args.lambda_W},  # Last layer
        {'params': (p for n, p in model.named_parameters() if n not in ['model.fc.weight', 'model.fc.bias']), 'weight_decay': 0}  
        ]
        optimizer = torch.optim.SGD(param_groups, lr=args.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.lambda_W)
    return optimizer

def main():
    # Parsing arguments 
    args = configs.get_args()

    # Set random seed
    set_seed(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegressionResNet(pretrained=True, num_outputs=args.y_dim)
    model = model.to(device)

    train_dataset = ImageTargetDataset('/vast/zz4330/Carla_JPG/Train/images', '/vast/zz4330/Carla_JPG/Train/targets', transform=transform)
    val_dataset = ImageTargetDataset('/vast/zz4330/Carla_JPG/Val/images', '/vast/zz4330/Carla_JPG/Val/targets', transform=transform)
    train_data_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=256, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = set_optimizer(args,model)
    
    train(model, train_data_loader, device, criterion, optimizer, args)


if __name__ == '__main__':
    main()