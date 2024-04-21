import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
import random

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
    metrics = {
        'embeddings': torch.empty((0,), device=device),
        'targets': torch.empty((0,), device=device),
        'outputs': torch.empty((0,), device=device),
        'weights': torch.empty((0,), device=device)
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
        loss.backward()
        optimizer.step()

        if metrics['embeddings'].size(0) < accum_size:
            embeddings = model.get_last_layer_embeddings(images)
            metrics['embeddings'] = torch.cat((metrics['embeddings'], embeddings.detach()), 0)
            metrics['targets'] = torch.cat((metrics['targets'], targets.detach()), 0)
            metrics['outputs'] = torch.cat((metrics['outputs'], outputs.detach()), 0)
        metrics['weights'] = model.model.fc.weight.detach()
        print(metrics['embeddings'].shape,metrics['targets'].shape,metrics['outputs'].shape, metrics['weights'].shape)
    return metrics

def train(model, train_data_loader, device, criterion, optimizer, args):
    all_results = {
        'cos_sim_y_Wh': [],
        'cos_sim_W': [],
        'cos_sim_H': [],
        'cos_sim_y_h_postPCA': [],
        'projection_error': [],
        'mse_cos_sim': [],
        'projection_error_H2W_E': []
    }
    for epoch in range(1, args.num_epochs + 1):
        metrics = train_epoch(model, epoch,train_data_loader, criterion, optimizer, device, args, accum_size=200 )
        result = calculate_metrics(metrics, device, args.y_dim)

        for key in all_results:
            all_results[key].append(result[key])

        plot_metrics_over_epochs(all_results, epoch, args.save_dir)
        if epoch % 10 ==0:
        
            save_model(epoch,model,optimizer,args.save_dir)
            
        print(f"Epoch {epoch+1}: Completed")

def calculate_metrics(metrics, device, y_dim):
    result = {}
    y = metrics['targets'].to(device)  #(B,N)
    Wh = metrics['outputs'].to(device) #(B,N)
    W = metrics['weights'].to(device) #(N,512)
    H = metrics['embeddings'].to(device) #(B,512)

    H_norm = F.normalize(H, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)

    # Cosine similarity calculations
    result['cos_sim_y_Wh'] = cosine_similarity_gpu(y,Wh).mean().item()
    result['cos_sim_W'] = cosine_similarity_gpu(W, W).fill_diagonal_(float('nan')).nanmean().item()
    result['cos_sim_H'] = cosine_similarity_gpu(H_norm, H_norm).fill_diagonal_(float('nan')).nanmean().item()

    # Convert to numpy for PCA due to lack of direct support in PyTorch
    H_norm_np = H_norm.cpu().detach().numpy()
    pca_for_H = PCA(n_components=y_dim)
    H_pca = pca_for_H.fit_transform(H_norm_np) 
    H_reconstruct = pca_for_H.inverse_transform(H_pca)
    result['projection_error'] = np.mean(np.square(H_norm_np - H_reconstruct))

    # Cosine similarity post PCA
    H_pca_norm = F.normalize(torch.tensor(H_pca).float().to(device), p=2, dim=1)
    cos_sim_after_pca = torch.mm(H_pca_norm, y_norm.transpose(0, 1))
    result['cos_sim_y_h_postPCA'] = cos_sim_after_pca.diag().mean().item()

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
    result['mse_cos_sim_pca'] = F.mse_loss(upper_tri_embeddings_pca, upper_tri_targets).item()

    # Projection error with Gram-Schmidt
    U = gram_schmidt(W)
    P_E = torch.mm(U.T, U)
    H_projected_E = torch.mm(H, P_E)
    result['projection_error_H2W_E'] = F.mse_loss(H_projected_E, H).item()
    print(result)
    return result

def plot_metrics_over_epochs(all_results, epoch, save_dir):
    # Plotting cosine similarities in one plot
    plt.figure(figsize=(10, 6))
    metrics_cosine = ['cos_sim_y_Wh', 'cos_sim_W', 'cos_sim_H', 'cos_sim_y_h_postPCA']
    for metric in metrics_cosine:
        plt.plot(range(1, epoch + 1), all_results[metric], label=metric)
    plt.title('Cosine Similarities Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/cosine_similarities_up_to_epoch_{epoch}.png")
    plt.close()

    # Plotting projection errors in one plot
    plt.figure(figsize=(10, 6))
    metrics_projection = ['projection_error', 'projection_error_H2W_E']
    for metric in metrics_projection:
        plt.plot(range(1, epoch + 1), all_results[metric], label=metric)
    plt.title('Projection Errors Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Projection Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/projection_errors_up_to_epoch_{epoch}.png")
    plt.close()

    # Plotting MSE cosine similarity in a single plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['mse_cos_sim'], label='MSE Cosine Similarity', color='magenta')
    plt.title('MSE of Cosine Similarities Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Cosine Similarity')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/mse_cosine_similarity_up_to_epoch_{epoch}.png")
    plt.close()

    print(f"Metrics plotted and saved up to epoch {epoch}")


