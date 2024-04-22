import argparse
import configs 
from torch.nn.functional import mse_loss
from dataset import ImageTargetDataset, transform,H5Dataset  
from model import RegressionResNet 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import train
import os
def print_memory_usage(description):
    print(f"{description}:")
    print(f"  Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"  Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
  
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
    print_memory_usage("Initial GPU Memory Usage")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegressionResNet(pretrained=True, num_outputs=args.y_dim)
    model = model.to(device)
    os.makedirs(args.save_dir, exist_ok=True)
    
    train_dataset = H5Dataset('/vast/zz4330/Carla_h5/SeqTrain', transform=transform)
    val_dataset = H5Dataset('/vast/zz4330/Carla_h5/SeqVal', transform=transform)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    print_memory_usage("Post-Dataset Loading GPU Memory Usage")

    criterion = nn.MSELoss()
    optimizer = set_optimizer(args,model)
    
    if args.load_from_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    train(model, train_data_loader, device, criterion, optimizer, args)


if __name__ == '__main__':
    main()
