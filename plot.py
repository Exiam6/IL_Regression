import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
import random
import json

def plot_metrics_over_epochs(all_results, all_results_valid, epoch, save_dir):
    
    # Plotting loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['loss'], label="Train", color='blue')
    plt.plot(range(1, epoch + 1), all_results_valid['loss'], label="Test", color='red')
    plt.title('Train and Test Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/Loss.png")
    plt.close()

    # Plotting W_norm
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['W_norm'], label="Train",color='blue')
    plt.plot(range(1, epoch + 1), all_results_valid['W_norm'], label="Test",color='red')
    plt.title('Train and Test W_norm Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('W_norm')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/W_norm.png")
    plt.close()

    # Plotting cosine similarities
    plt.figure(figsize=(10, 6))
    metrics_cosine = ['cos_sim_y_Wh', 'cos_sim_W', 'cos_sim_H', 'cos_sim_y_h_postPCA','cos_sim_y_h_H2W_E']
    colors = ['blue', 'green', 'red', 'purple', 'orange'] 
    for metric, color in zip(metrics_cosine, colors):
        plt.plot(range(1, epoch + 1), all_results[metric], label=metric, color=color)
    plt.title('Train Cosine Similarities Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/cosine_similarities_train.png")
    plt.close()

    # Plotting cosine similarities
    plt.figure(figsize=(10, 6))
    metrics_cosine = ['cos_sim_y_Wh', 'cos_sim_W', 'cos_sim_H', 'cos_sim_y_h_postPCA','cos_sim_y_h_H2W_E']
    for metric, color in zip(metrics_cosine, colors):
        plt.plot(range(1, epoch + 1), all_results_valid[metric], label=metric, color=color)
    plt.title('Test Cosine Similarities Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/cosine_similarities_test.png")
    plt.close()

    # Plotting projection errors in one plot
    plt.figure(figsize=(10, 6))
    metrics_projection = ['projection_error_PCA', 'projection_error_H2W_E']
    train_colors = ['cyan', 'magenta']  
    test_colors = ['darkcyan', 'darkmagenta']  
    for metric, color in zip(metrics_projection, train_colors):
        plt.plot(range(1, epoch + 1), all_results[metric], label=f"Train {metric}", color=color)
    for metric, color in zip(metrics_projection, test_colors):
        plt.plot(range(1, epoch + 1), all_results_valid[metric], label=f"Test {metric}", color=color)
    plt.title('Projection Errors Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Projection Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/projection_errors_up_to_epoch.png")
    plt.close()

    # Plotting MSE cosine similarity
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['mse_cos_sim'], label='Train_H2W', color='magenta')
    plt.plot(range(1, epoch + 1), all_results_valid['mse_cos_sim'], label='Test_H2W', color='darkmagenta')
    plt.plot(range(1, epoch + 1), all_results['mse_cos_sim_PCA'], label='Train_PCA', color='cyan')
    plt.plot(range(1, epoch + 1), all_results_valid['mse_cos_sim_PCA'], label='Test_PCA',color='darkcyan')
    
    plt.title('MSE of Cosine Similarities Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Cosine Similarity')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/mse_cosine_similarity_up_to_epoch.png")
    plt.close()

    print(f"Metrics plotted and saved up to epoch")
