import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--save_dir', type=str, default="/scratch/zz4330/IL_Regression_2D_V2/Result/2DCase2_W5e-2H1e-5", help='Directory to save results')
    parser.add_argument('--checkpoint_path', type=str, default="/scratch/zz4330/IL_Regression/Result/Case2_W5e-2H5e-5/checkpoints/model_checkpoint_epoch_160.pth", help='Path to model checkpoint')
    parser.add_argument('--case2', default=True, action='store_true', help='Enable case 2 scenario')
    parser.add_argument('--y_dim', type=int, default=2, help='Dimension of output')
    parser.add_argument('--debug', default=False, help='Only runs 20 batches per epoch for debugging, and set debug to true', action='store_true')
    
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lambda_H', type=float, default=1e-5, help='Regularization lambda H')
    parser.add_argument('--lambda_W', type=float, default=5e-2, help='Regularization lambda W')
    
    parser.add_argument('--train_data_path', type=str, default='/scratch/zz4330/Carla/Train/images.npy', help='Training data path')
    parser.add_argument('--train_target_path', type=str, default='/scratch/zz4330/Carla/Train/targets.npy', help='Training target path')
    parser.add_argument('--val_data_path', type=str, default='/scratch/zz4330/Carla/Val/images.npy', help='Validation data path')
    parser.add_argument('--val_target_path', type=str, default='/scratch/zz4330/Carla/Val/targets.npy', help='Validation target path')
   
    parser.add_argument('--load_from_checkpoint',  default=False,action='store_true', help='Load model from a checkpoint')
    
    parser.add_argument('--sampling_rate', type=float, default=0.1, help='Sampling rate for training')
    parser.add_argument('--start_epoch', type=int, default=81, help='Starting epoch number')
    parser.add_argument('--random_seed', type=int, default=43, help='Random seed')
    return parser.parse_args()

def main():
    args = get_args()

    print(f"Device: {args.device}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Save directory: {args.save_dir}")
    print(f"Debug mode: {args.debug}")


if __name__ == "__main__":
    main()

