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

def to_serializable(val):
    """Converts NumPy arrays and PyTorch tensors to Python lists; converts NumPy and native Python numeric types to Python floats."""
    if isinstance(val, np.ndarray):   
        return val.tolist()  
    elif isinstance(val, torch.Tensor):   
        return val.tolist()   
    elif isinstance(val, (np.number, int, float)):   
        return float(val)   
    elif isinstance(val, dict):   
        return {k: to_serializable(v) for k, v in val.items()}
    return val   

def save_results(results, results_valid, save_dir, filename='results.json'):
    path = os.path.join(save_dir, filename)
    data = {
        'training_results': results,
        'validation_results': results_valid
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, default=to_serializable)

    print(f"Results saved to {path}")

def normalize_tensor(tensor):
    
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True)
    return (tensor - mean) / std

def gram_schmidt(W):
    for i in range(W.shape[0]):
        for j in range(i):
            W[i,:] -= torch.dot(W[i,:],W[j,:]) * W[j,:]
        W[i, :] = W[i, :] / torch.norm(W[i, :], p=2)
    return W

def cosine_similarity_gpu(a, b):
  
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)

    return torch.mm(a_norm, b_norm.transpose(0,1))


def save_model(epoch,model,optimizer, save_dir):
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }
    checkpoint_path = os.path.join(save_dir, f'checkpoints/model_checkpoint_epoch_{epoch}.pth')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f'Model checkpoint saved at {checkpoint_path}')

def train_epoch(model, epoch, train_data_loader, criterion, optimizer, device, args, accum_size):
    model.train()
    train_count = 0
    total_loss=0
    metrics = {
        'embeddings': torch.empty((0,), device=device),
        'targets': torch.empty((0,), device=device),
        'outputs': torch.empty((0,), device=device),
        'weights': torch.empty((0,), device=device),
        'loss':0,
    }
    
    for batch in tqdm(train_data_loader, desc=f"Epoch {epoch}/{args.num_epochs} Training"):
        train_count+=1
        if args.debug:
            if train_count>10:
                break
        images = batch['image'].to(device)
        targets = batch['target'].to(device)
  
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        embeddings = model.get_last_layer_embeddings(images)
        l2reg_H = args.lambda_H * torch.norm(embeddings, 2)
        #l2reg_W = lambda_W * torch.norm(last_layer_weights, 2)
        if args.case2:
            loss = loss + l2reg_H
        total_loss += loss.item() 
        loss.backward()
        optimizer.step()

        if train_count  < accum_size:
            embeddings = model.get_last_layer_embeddings(images)
            metrics['embeddings'] = torch.cat((metrics['embeddings'], embeddings.detach()), 0)
            metrics['targets'] = torch.cat((metrics['targets'], targets.detach()), 0)
            metrics['outputs'] = torch.cat((metrics['outputs'], outputs.detach()), 0)
        metrics['weights'] = model.model.fc.weight.detach()
       
    metrics['loss'] = total_loss/train_count
    return metrics


def validate_epoch(model, epoch, val_data_loader, criterion, optimizer, device, args, accum_size):
    model.eval()
    count = 0
    total_loss=0
    metrics = {
        'embeddings': torch.empty((0,), device=device),
        'targets': torch.empty((0,), device=device),
        'outputs': torch.empty((0,), device=device),
        'weights': torch.empty((0,), device=device),
        'loss':0
    }
    
    for batch in tqdm(val_data_loader, desc=f"Epoch {epoch}/{args.num_epochs} Validation"):
        count+=1
        if args.debug:
            if count>10:
                break
        images = batch['image'].to(device)
        targets = batch['target'].to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item() 
        print(images.shape)
        if count < accum_size:
            embeddings = model.get_last_layer_embeddings(images)
            metrics['embeddings'] = torch.cat((metrics['embeddings'], embeddings.detach()), 0)
            metrics['targets'] = torch.cat((metrics['targets'], targets.detach()), 0)
            metrics['outputs'] = torch.cat((metrics['outputs'], outputs.detach()), 0)
    metrics['weights'] = model.model.fc.weight.detach()
    metrics['loss'] = total_loss/count
    return metrics

def train(model, train_data_loader, device, criterion, optimizer, args):
    all_results = {
        'cos_sim_y_Wh': [],
        'cos_sim_W': [],
        'cos_sim_H': [],
        'cos_sim_y_h_postPCA': [],
        'cos_sim_y_h_H2W_E': [],
        'projection_error_PCA': [],
        'projection_error_H2W_E': [],
        'mse_cos_sim': [],
        'mse_cos_sim_PCA': [],
        'loss':[],
        'W_norm':[],
    }
    all_results_valid = {
        'cos_sim_y_Wh': [],
        'cos_sim_W': [],
        'cos_sim_H': [],
        'cos_sim_y_h_postPCA': [],
        'cos_sim_y_h_H2W_E': [],
        'projection_error_PCA': [],
        'projection_error_H2W_E': [],
        'mse_cos_sim': [],
        'mse_cos_sim_PCA': [],
        'loss':[],
        'W_norm':[],
    }
    for epoch in range(1, args.num_epochs + 1):
        metrics = train_epoch(model, epoch,train_data_loader, criterion, optimizer, device, args, accum_size=200 )
        metrics_valid = validate_epoch(model, epoch,train_data_loader, criterion, optimizer, device, args, accum_size=100 )
       
        result = calculate_metrics(metrics, device, args.y_dim)
        result_valid = calculate_metrics(metrics_valid, device, args.y_dim)
        for key in all_results:
            all_results[key].append(result[key])
        for key in all_results_valid:
            all_results_valid[key].append(result_valid[key])
        print(all_results,all_results_valid)
        plot_metrics_over_epochs(all_results, all_results_valid, epoch, args.save_dir)

        if epoch % 10 ==0:
            save_model(epoch,model,optimizer,args.save_dir)
            save_results(all_results, all_results_valid, args.save_dir, filename=f'results.json')

        print(f"Epoch {epoch+1}: Completed")

def calculate_metrics(metrics, device, y_dim):
    result = {}
    y = metrics['targets'].to(device)  #(B,2)
    Wh = metrics['outputs'].to(device) #(B,2)
    W = metrics['weights'].to(device) #(2,512)
    H = metrics['embeddings'].to(device) #(B,512)

    H_norm = F.normalize(H, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    W_norm = F.normalize(W, p=2, dim=1)
    result['W_norm'] = torch.norm(W, p=2).item()
    result['loss'] = metrics['loss']
    # Cosine similarity calculations
    result['cos_sim_y_Wh'] = cosine_similarity_gpu(y,Wh).mean().item()
    result['cos_sim_W'] = cosine_similarity_gpu(W, W).fill_diagonal_(float('nan')).nanmean().item()
    result['cos_sim_H'] = cosine_similarity_gpu(H_norm, H_norm).fill_diagonal_(float('nan')).nanmean().item()

    # H with PCA
    H_norm_np = H_norm.cpu().detach().numpy()
    pca_for_H = PCA(n_components=y_dim)
    H_pca = pca_for_H.fit_transform(H_norm_np) 
    H_reconstruct = pca_for_H.inverse_transform(H_pca)
    result['projection_error_PCA'] = np.mean(np.square(H_norm_np - H_reconstruct))

    # Cosine similarity of Y and H post PCA
    H_pca_norm = F.normalize(torch.tensor(H_pca).float().to(device), p=2, dim=1)
    cos_sim_y_h_after_pca = torch.mm(H_pca_norm, y_norm.transpose(0, 1))
    result['cos_sim_y_h_postPCA'] = cos_sim_y_h_after_pca.diag().mean().item()

    # MSE between cosine similarities of embeddings and targets
    cos_H = torch.mm(H_norm, H_norm.transpose(0, 1))
    cos_y = torch.mm(y_norm, y_norm.transpose(0, 1))
    indices = torch.triu_indices(cos_H.size(0), cos_H.size(0), offset=1)
    upper_tri_embeddings = cos_H[indices[0], indices[1]]
    upper_tri_targets = cos_y[indices[0], indices[1]]
    result['mse_cos_sim'] = F.mse_loss(upper_tri_embeddings, upper_tri_targets).item()

    # MSE between cosine similarities of PCA embeddings and targets
    cos_H_pca = torch.mm(H_pca_norm, H_pca_norm.transpose(0, 1))
    indices = torch.triu_indices(cos_H_pca.size(0), cos_H_pca.size(0), offset=1)
    upper_tri_embeddings_pca = cos_H_pca[indices[0], indices[1]]
    result['mse_cos_sim_PCA'] = F.mse_loss(upper_tri_embeddings_pca, upper_tri_targets).item()

    # Projection error with Gram-Schmidt
    U = gram_schmidt(W)
    P_E = torch.mm(U.T, U)
    H_projected_E = torch.mm(H, P_E)
    result['projection_error_H2W_E'] = F.mse_loss(H_projected_E, H).item()

    # Cosine similarity of Y and H with H2W
    H_coordinates = torch.mm(H, U.T)
    H_coordinates_norm = F.normalize(torch.tensor(H_coordinates).float().to(device), p=2, dim=1)
    cos_sim_H2W = torch.mm(H_coordinates_norm, y_norm.transpose(0, 1))
    result['cos_sim_y_h_H2W_E'] = cos_sim_H2W.diag().mean().item()

    print(result)
    return result

def plot_metrics_over_epochs(all_results, all_results_valid, epoch, save_dir):
    
    # Plotting loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['loss'], label="Train", color='blue')
    plt.plot(range(1, epoch + 1), all_results_valid['loss'], label="Test", color='red')
    plt.title('Train and Test W_norm Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('W_norm')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/W_norm.png")
    plt.close()

    # Plotting W_norm
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['W_norm'], label="Train",color='blue')
    plt.plot(range(1, epoch + 1), all_results_valid['W_norm'], label="Train",color='red')
    plt.title('Train and Test Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/Loss.png")
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
    plt.plot(range(1, epoch + 1), all_results['mse_cos_sim'], label='Train ', color='magenta')
    plt.plot(range(1, epoch + 1), all_results_valid['mse_cos_sim'], label='Test', color='darkmagenta')
    plt.plot(range(1, epoch + 1), all_results['mse_cos_sim_PCA'], label='Train ', color='cyan')
    plt.plot(range(1, epoch + 1), all_results_valid['mse_cos_sim_PCA'], label='Test',color='darkcyan')
    
    plt.title('MSE of Cosine Similarities Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Cosine Similarity')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/mse_cosine_similarity_up_to_epoch.png")
    plt.close()

    print(f"Metrics plotted and saved up to epoch")



