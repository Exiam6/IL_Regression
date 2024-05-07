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
from plot import plot_metrics_over_epochs
import math

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
    U = torch.empty_like(W)
    U[0, :] = W[0, :] / torch.norm(W[0, :], p=2)

    proj = torch.dot(U[0, :], W[1, :]) * U[0, :]
    ortho_vector = W[1, :] - proj
    U[1, :] = ortho_vector / torch.norm(ortho_vector, p=2)

    return U

def cosine_similarity_gpu(a, b):
  
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)

    return torch.mm(a_norm, b_norm.T)


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
        l2reg_H = args.lambda_H * (torch.norm(embeddings, 2)**2)/args.batch_size
        l2reg_W = args.lambda_W * (torch.norm(model.model.fc.weight, 2)**2)
        if args.case2:
            loss = loss + l2reg_H + l2reg_W
        total_loss += loss.item() 
        loss.backward()
        optimizer.step()
        if train_count>100:
            break

    metrics['loss'] = total_loss/(train_count*args.batch_size)
    return metrics


def check_epoch(model, epoch, val_data_loader, criterion, optimizer, device, args, accum_size):
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
        images = batch['image'].to(device) #torch.Size([48, 3, 200, 88])
        targets = batch['target'].to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item() 
        if count < accum_size:
            embeddings = model.get_last_layer_embeddings(images)
            metrics['embeddings'] = torch.cat((metrics['embeddings'], embeddings.detach()), 0)
            metrics['targets'] = torch.cat((metrics['targets'], targets.detach()), 0)
            metrics['outputs'] = torch.cat((metrics['outputs'], outputs.detach()), 0)
        else:
            break
    metrics['weights'] = model.model.fc.weight.detach()
    metrics['loss'] = total_loss/(count*args.batch_size)
    return metrics

def train(model, train_data_loader,val_data_loader, device, criterion, optimizer, args):
    all_results_train = {
        'cos_sim_y_Wh': [],
        'W11_product': [],
        'W12_product': [],
        'W22_product': [],
        'W11_norm_square_theory':[],
        'W12_norm_square_theory':[],
        'W22_norm_square_theory':[],
        'W11_Cov_sqrt':[],
        'W12_Cov_sqrt':[],
        'W22_Cov_sqrt':[],
        'W11_nc2':[],
        'W12_nc2':[],
        'W22_nc2':[],
        'cos_sim_y': [],
        'cos_sim_W': [],
        'cos_sim_H': [],
        'cos_sim_y_h_postPCA': [],
        'cos_sim_y_h_H2W_E': [],
        'projection_error_PCA': [],
        'projection_error_H2W_E': [],
        'mse_cos_sim': [],
        'mse_cos_sim_norm': [],
        'loss':[],
        'NC2':[],
        'W_norm_square':[],
    }
    all_results_valid = {
        'cos_sim_y_Wh': [],
        'W11_product': [],
        'W12_product': [],
        'W22_product': [],
        'W11_norm_square_theory':[],
        'W12_norm_square_theory':[],
        'W22_norm_square_theory':[],
        'W11_Cov_sqrt':[],
        'W12_Cov_sqrt':[],
        'W22_Cov_sqrt':[],
        'W11_nc2':[],
        'W12_nc2':[],
        'W22_nc2':[],
        'cos_sim_y': [],
        'cos_sim_W': [],
        'cos_sim_H': [],
        'cos_sim_y_h_postPCA': [],
        'cos_sim_y_h_H2W_E': [],
        'projection_error_PCA': [],
        'projection_error_H2W_E': [],
        'mse_cos_sim': [],
        'mse_cos_sim_norm': [],
        'loss':[],
        'NC2':[],
        'W_norm_square':[],
    }
    for epoch in range(1, args.num_epochs + 1):
        metric_when_training = train_epoch(model, epoch,train_data_loader, criterion, optimizer, device, args, accum_size=10)
        metrics_train = check_epoch(model, epoch,train_data_loader, criterion, optimizer, device, args, accum_size=10)
        metrics_train['loss']= metric_when_training['loss']
        metrics_valid = check_epoch(model, epoch,val_data_loader, criterion, optimizer, device, args, accum_size=10)
       
        result_train = calculate_metrics(metrics_train, device,epoch, args)
        result_valid = calculate_metrics(metrics_valid, device,epoch, args)
        for key in all_results_train:
            all_results_train[key].append(result_train[key])
        for key in all_results_valid:
            all_results_valid[key].append(result_valid[key])
        plot_metrics_over_epochs(all_results_train, all_results_valid, epoch, args.save_dir)

        if epoch % 10 ==0:
            save_model(epoch,model,optimizer,args.save_dir)
            save_results(all_results_train, all_results_valid, args.save_dir, filename=f'results.json')

        print(f"Epoch {epoch+1}: Completed")

def find_c(W, Sigma_sqrt, args,device):
    min_diff = float('inf')
    optimal_c = 0
    lambda_min = torch.min(torch.linalg.eigvalsh(Sigma_sqrt)).item() 
    I = torch.eye(args.y_dim, device=device)
    WW = W @ W.T
    norm_WW = torch.norm(WW, p='fro')
    for c in torch.linspace(0, lambda_min, 1000):  
        c_sqrt_tensor = torch.sqrt(c).to(device)
        Sigma_mod = Sigma_sqrt - c_sqrt_tensor * I
        norm_Sigma_mod = torch.norm(Sigma_mod, p='fro')
        
        diff = torch.norm(WW / norm_WW - Sigma_mod / norm_Sigma_mod, p='fro').item()
        
        if diff < min_diff:
            min_diff = diff
            optimal_c = c
    
    return optimal_c, min_diff

def calculate_metrics(metrics, device,epoch, args):
    result = {}
    y = metrics['targets'].to(device)  #(B,2)
    Wh = metrics['outputs'].to(device) #(B,2)
    W = metrics['weights'].to(device) #(2,512)
    H = metrics['embeddings'].to(device) #(B,512)
    print("Y",y.shape)
    print("Wh",Wh.shape)
    print("W",W.shape)
    print("W0",W[0].shape)
    print("H",H.shape)
    H_norm = F.normalize(H, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    W_norm = F.normalize(W, p=2, dim=1)
    result['W_norm_square'] = (torch.norm(W, p=2).item())**2

    Sigma = torch.matmul(y.T, y) / y.size(0)
    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    Sigma_sqrt = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T

    W_norm_square_theory = args.lambda_H * (Sigma_sqrt / torch.sqrt(torch.tensor(args.lambda_H * args.lambda_W, device=device)) - torch.eye(args.y_dim, device=device))
    min_c,result['NC2'] = find_c(W, Sigma_sqrt, args,device)
    del Sigma,eigenvalues, eigenvectors,sqrt_eigenvalues

    Sigma_sqrt_inverse = torch.linalg.inv(Sigma_sqrt)
    y_delta = torch.mm(y,Sigma_sqrt_inverse) 

    result['W11_norm_square_theory'] = W_norm_square_theory[0,0].item()
    result['W12_norm_square_theory'] = W_norm_square_theory[0,1].item()
    result['W22_norm_square_theory'] = W_norm_square_theory[1,1].item()
    result['W11_Cov_sqrt'] = Sigma_sqrt[0,0].item()
    result['W12_Cov_sqrt'] = Sigma_sqrt[0,1].item()
    result['W22_Cov_sqrt'] = Sigma_sqrt[1,1].item()
    result['loss'] = metrics['loss']
    # Cosine similarity calculations
    

    result['cos_sim_y_Wh'] = cosine_similarity_gpu(y,Wh).mean().item()
    result['W11_product'] = torch.dot(W[0], W[0]).item()
    result['W12_product'] = torch.dot(W[0], W[1]).item()
    result['W22_product'] = torch.dot(W[1], W[1]).item()
    result['W11_nc2'] = result['W11_product']-result['W11_Cov_sqrt']
    result['W12_nc2'] = result['W12_product']-result['W12_Cov_sqrt']
    result['W22_nc2'] = result['W22_product']-result['W22_Cov_sqrt']

    result['cos_sim_W'] = cosine_similarity_gpu(W, W).fill_diagonal_(float('nan')).nanmean().item()
    result['cos_sim_H'] = cosine_similarity_gpu(H, H).fill_diagonal_(float('nan')).nanmean().item()
    result['cos_sim_y'] = cosine_similarity_gpu(y, y).fill_diagonal_(float('nan')).nanmean().item()

    # H with PCA
    H_norm_np = H.cpu().detach().numpy()
    pca_for_H = PCA(n_components=args.y_dim)
    H_pca = pca_for_H.fit_transform(H_norm_np) 
    H_reconstruct = pca_for_H.inverse_transform(H_pca)
    result['projection_error_PCA'] = np.mean(np.square(H_norm_np - H_reconstruct))

    # Cosine similarity of Y and H post PCA
    H_pca_norm = F.normalize(torch.tensor(H_pca).float().to(device), p=2, dim=1)
    cos_sim_y_h_after_pca = torch.mm(H_pca_norm, y_norm.T)
    result['cos_sim_y_h_postPCA'] = cos_sim_y_h_after_pca.diag().mean().item()

    # MSE between cosine similarities of embeddings and targets with norm
    cos_H_norm = torch.mm(H_norm, H_norm.T)
    cos_y_norm = torch.mm(y_norm, y_norm.T)
    indices = torch.triu_indices(cos_H_norm.size(0), cos_H_norm.size(0), offset=1)
    upper_tri_embeddings_norm = cos_H_norm[indices[0], indices[1]]
    upper_tri_targets_norm = cos_y_norm[indices[0], indices[1]]
    result['mse_cos_sim_norm'] = F.mse_loss(upper_tri_embeddings_norm, upper_tri_targets_norm).item()

    # MSE between cosine similarities of embeddings and targets
    cos_H = torch.mm(H, H.transpose(0, 1))
    cos_y = torch.mm(y, y.transpose(0, 1))
    indices = torch.triu_indices(cos_H.size(0), cos_H.size(0), offset=1)
    upper_tri_embeddings = cos_H[indices[0], indices[1]]
    upper_tri_targets = cos_y[indices[0], indices[1]]
    result['mse_cos_sim'] = F.mse_loss(upper_tri_embeddings, upper_tri_targets).item()


    # MSE between cosine similarities of PCA embeddings and targets
    #cos_H_pca = torch.mm(H_pca_norm, H_pca_norm.transpose(0, 1))
    #indices = torch.triu_indices(cos_H_pca.size(0), cos_H_pca.size(0), offset=1)
    #upper_tri_embeddings_pca = cos_H_pca[indices[0], indices[1]]
    #result['mse_cos_sim_PCA'] = F.mse_loss(upper_tri_embeddings_pca, upper_tri_targets).item()

    # Projection error with Gram-Schmidt
    U = gram_schmidt(W)
    P_E = torch.mm(U.T, U)
    H_projected_E = torch.mm(H, P_E)
    #H_projected_E_norm = F.normalize(torch.tensor(H_projected_E).float().to(device), p=2, dim=1)
    result['projection_error_H2W_E'] = F.mse_loss(H_projected_E, H).item()
    os.makedirs(f"{args.save_dir}figs", exist_ok=True)
    # Cosine similarity of Y and H with H2W
    H_coordinates = torch.mm(H_norm, U.T)
    H_coordinates_norm = F.normalize(H_coordinates.clone().detach().to(device), p=2, dim=1)
    cos_sim_H2W = torch.mm(H_coordinates_norm, y_norm.T)
    result['cos_sim_y_h_H2W_E'] = cos_sim_H2W.diag().mean().item()

    H_coordinates = H_coordinates.cpu().numpy()
    H_coordinates_norm = H_coordinates_norm.cpu().numpy()
    plt.figure(figsize=(8, 6))
    colors = y_delta[:, 0] / y_delta[:, 1]
    colors=colors.cpu().numpy()
    vmin, vmax = np.percentile(colors, [1, 50])
    scatter = plt.scatter(H_coordinates[:, 0], H_coordinates[:, 1], c=colors, cmap='viridis',vmin=vmin, vmax=vmax)
    plt.colorbar(scatter, label='y1/y2 ratio')
    plt.xlabel('W1')
    plt.ylabel('W2')
    plt.title('2D Coordinates Colored by y1/y2 Ratio')
    plt.grid(True)
    plt.show()
    plt.savefig(f"{args.save_dir}figs/{epoch}_y_fig1.png")


    return result

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
    plt.savefig(f"{save_dir}Loss.png")
    plt.close()

    # Plotting W_norm
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['W_norm_square'], label="Train",color='blue')
    plt.plot(range(1, epoch + 1), all_results_valid['W_norm_square'], label="Test",color='red')
    plt.title('Train and Test W_norm_square Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('W_norm_square')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}W_norm_square.png")
    plt.close()


    # Plotting cosine similarities for W
    plt.figure(figsize=(10, 6))
    metrics_cosine = ['W11_product', 'W12_product', 'W22_product', 'W11_norm_square_theory','W12_norm_square_theory','W22_norm_square_theory']
    colors = ['#00008B', '#006400','#8B0000','#ADD8E6' ,'#90EE90', '#FFA07A']  # Hex codes for darkblue, lightblue, darkgreen, lightgreen, darkred, light salmon (a light red)
    for metric, color in zip(metrics_cosine, colors):
        plt.plot(range(1, epoch + 1), all_results[metric], label=metric, color=color)
    plt.title('Train W_Matrix Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('W_Matrix')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}W_Matrix_train.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    metrics_cosine = ['W11_product',  'W11_norm_square_theory']
    colors = ['#00008B','#ADD8E6' ] 
    for metric, color in zip(metrics_cosine, colors):
        plt.plot(range(1, epoch + 1), all_results[metric], label=metric, color=color)
    plt.title('Train W11_Matrix Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('W11_Matrix')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}W11_Matrix_train.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    metrics_cosine = [ 'W12_product', 'W12_norm_square_theory']
    colors = ['#006400','#90EE90'] 
    for metric, color in zip(metrics_cosine, colors):
        plt.plot(range(1, epoch + 1), all_results[metric], label=metric, color=color)
    plt.title('Train W12_Matrix Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('W12_Matrix')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}W12_Matrix_train.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    metrics_cosine = ['W22_product','W22_norm_square_theory']
    colors = ['#8B0000', '#FFA07A']  
    for metric, color in zip(metrics_cosine, colors):
        plt.plot(range(1, epoch + 1), all_results[metric], label=metric, color=color)
    plt.title('Train W22_Matrix Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('W22_Matrix')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/W22_Matrix_train.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['W11_nc2'], label='W11_NC2', color='blue')
    plt.title('Train W11_NC2 Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('W11_Matrix')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}W11_NC2_train.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['W12_nc2'], label='W12_NC2', color='blue')
    plt.title('Train W12_NC2 Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('W12_Matrix')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}W12_NC2_train.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['W22_nc2'], label='W22_NC2', color='blue')
    plt.title('Train W22_NC2 Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('W22_Matrix')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}W22_NC2_train.png")
    plt.close()


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['NC2'], label='NC2', color='blue')
    plt.title('Train NC2 Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Best NC2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}NC2.png")
    plt.close()

    # Plotting cosine similarities
    plt.figure(figsize=(10, 6))
    metrics_cosine = ['cos_sim_y_Wh', 'cos_sim_W', 'cos_sim_H','cos_sim_y','cos_sim_y_h_postPCA','cos_sim_y_h_H2W_E']
    colors = ['blue', 'green', 'red', 'purple', 'orange','yellow'] 
    for metric, color in zip(metrics_cosine, colors):
        plt.plot(range(1, epoch + 1), all_results[metric], label=metric, color=color)
    plt.title('Train Cosine Similarities Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}cosine_similarities_train.png")
    plt.close()

    # Plotting cosine similarities
    plt.figure(figsize=(10, 6))
    metrics_cosine = ['cos_sim_y_Wh', 'cos_sim_W', 'cos_sim_H','cos_sim_y','cos_sim_y_h_postPCA','cos_sim_y_h_H2W_E']
    colors = ['blue', 'green', 'red', 'purple', 'orange','yellow'] 
    for metric, color in zip(metrics_cosine, colors):
        plt.plot(range(1, epoch + 1), all_results_valid[metric], label=metric, color=color)
    plt.title('Test Cosine Similarities Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}cosine_similarities_test.png")
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
    plt.savefig(f"{save_dir}projection_errors_up_to_epoch.png")
    plt.close()

    # Plotting MSE cosine similarity
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['mse_cos_sim'], label='Train ', color='magenta')
    plt.plot(range(1, epoch + 1), all_results_valid['mse_cos_sim'], label='Test', color='darkmagenta')
    plt.plot(range(1, epoch + 1), all_results['mse_cos_sim_norm'], label='Train_norm', color='cyan')
    plt.plot(range(1, epoch + 1), all_results_valid['mse_cos_sim_norm'], label='Test_norm',color='darkcyan')
    plt.title('MSE of Cosine Similarities Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Cosine Similarity')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}mse_cosine_similarity_up_to_epoch.png")
    plt.close()

    print(f"Metrics plotted and saved up to epoch")

