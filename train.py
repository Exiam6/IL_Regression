import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
import random
import json
from plot import plot_metrics_over_epochs
import math
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
def to_serializable(val):
   
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

def train_epoch(model, epoch, train_data_loader, criterion, optimizer, device, args, warmup_scheduler,accum_size):
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
        l2reg_H = args.lambda_W * (torch.norm(embeddings, 2)**2)/args.batch_size
        l2reg_W = args.lambda_W * (torch.norm(model.fc.weight, 2)**2)
        if args.case2:
            loss = loss + l2reg_H + l2reg_W
        total_loss += loss.item() 
        loss.backward()
        optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()
        metrics['embeddings'] = torch.cat((metrics['embeddings'], embeddings.detach()), 0)

    metrics['loss'] = total_loss/(train_count)
    return metrics


def get_all_y(val_data_loader, device):

    count = 0
    total_loss=0
    metrics = {
        'targets': torch.empty((0,), device=device),
        
    }
    
    for batch in tqdm(val_data_loader):
        count+=1

        targets = batch['target'].to(device)
        metrics['targets'] = torch.cat((metrics['targets'], targets.detach()), 0)
   
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
            if count>3:
                break
        images = batch['image'].to(device) #torch.Size([48, 3, 200, 88])
        targets = batch['target'].to(device)
        outputs= model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item() 
        if count < accum_size:
            embeddings = model.get_last_layer_embeddings(images)
            metrics['embeddings'] = torch.cat((metrics['embeddings'], embeddings.detach()), 0)
            metrics['targets'] = torch.cat((metrics['targets'], targets.detach()), 0)
            metrics['outputs'] = torch.cat((metrics['outputs'], outputs.detach()), 0)
        else:
            break
    metrics['weights'] = model.fc.weight.detach()
    metrics['loss'] = total_loss/(count)
    return metrics

def train(model, train_data_loader,val_data_loader, device, criterion, optimizer, args):
    all_results_train = {
        'cos_sim_y_Wh': [],
        'W11_product': [],
        'W12_product': [],
        'W22_product': [],
        'WW11_norm': [],
        'WW12_norm': [],
        'WW22_norm': [],
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
        'NRC1': [],
        'NRC1N': [],
        'NRC1_1': [],
        'NRC1_3': [],
        'NRC1_4': [],
        'NRC1_5': [],
        'mse_cos_sim': [],
        'mse_cos_sim_norm': [],
        'loss':[],
        'NRC2':[],
        'NRC2N': [],
        'NRC3':[],
        'C':[],
        'NC2_K':[],
        'K':[],
        'W_norm_square':[],
        'H_W_angles0':[],
        'H_W_angles1':[],
        'R2_score':[],
        'Explained_PCA_ratio1':[],
        'Explained_PCA_ratio2':[],
        'Explained_PCA_ratio3':[],
        'Explained_PCA_ratio4':[],
        'Explained_PCA_ratio5':[],
        'norm_H':[]
    }
    all_results_valid = {
        'cos_sim_y_Wh': [],
        'W11_product': [],
        'W12_product': [],
        'W22_product': [],
        'WW11_norm': [],
        'WW12_norm': [],
        'WW22_norm': [],
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
        'NRC1': [],
        'NRC1N': [],
        'NRC1_1': [],
        'NRC1_3': [],
        'NRC1_4': [],
        'NRC1_5': [],
        'mse_cos_sim': [],
        'mse_cos_sim_norm': [],
        'loss':[],
        'NRC2':[],
        'NRC2N': [],
        'NRC3':[],
        'C':[],
        'NC2_K':[],
        'K':[],
        'W_norm_square':[],
        'H_W_angles0':[],
        'H_W_angles1':[],
        'R2_score':[],
        'Explained_PCA_ratio1':[],
        'Explained_PCA_ratio2':[],
        'Explained_PCA_ratio3':[],
        'Explained_PCA_ratio4':[],
        'Explained_PCA_ratio5':[],
        'norm_H':[]
    }

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / 5))
    step_scheduler = MultiStepLR(optimizer, milestones=[30,60],gamma=0.2)

    for epoch in range(1, args.num_epochs + 1):
        
        metric_when_training = train_epoch(model, epoch,train_data_loader, criterion, optimizer, device, args,warmup_scheduler, accum_size=10)
        metrics_train = check_epoch(model, epoch,train_data_loader, criterion, optimizer, device, args, accum_size=10)
        #metrics_train['embeddings']= metric_when_training['embeddings']
        metrics_train['loss']= metric_when_training['loss']
        metrics_valid = check_epoch(model, epoch,val_data_loader, criterion, optimizer, device, args, accum_size=10)
        y_metrics=get_all_y(train_data_loader, device)
        result_train = calculate_metrics(metrics_train, device,epoch, args,y_metrics)
        result_valid = calculate_metrics(metrics_valid, device,epoch, args,y_metrics)
        for key in all_results_train:
            all_results_train[key].append(result_train[key])
        for key in all_results_valid:
            all_results_valid[key].append(result_valid[key])
        plot_metrics_over_epochs(all_results_train, all_results_valid, epoch, args.save_dir)
        step_scheduler.step()

        if epoch % 10 ==0:
            save_model(epoch,model,optimizer,args.save_dir)
            save_results(all_results_train, all_results_valid, args.save_dir, filename=f'results.json')

        print(f"Epoch {epoch+1}: Completed")

def find_c(W, Sigma_sqrt, args,device):
    min_diff = float('inf')
    optimal_c = 0
    lambda_min = torch.min(torch.linalg.eigvalsh(Sigma_sqrt)).item()  # smallest eigenvalue of Sigma_sqrt
    norm_WW = torch.norm(W @ W.T, p='fro')
    c_list=[]
    diff_list=[]
    for c in torch.linspace(lambda_min*0.001, lambda_min*1.001, 1000):  
        c_sqrt_tensor = torch.sqrt(c).to(device)
        I = torch.eye(args.y_dim, device=device)
        Sigma_mod = Sigma_sqrt - c_sqrt_tensor * I
        norm_Sigma_mod = torch.norm(Sigma_mod, p='fro')
        
        diff = torch.norm((W @ W.T) / norm_WW - Sigma_mod / norm_Sigma_mod, p='fro').item()
        c_list.append(c)
        diff_list.append(diff)
        if diff < min_diff:
            min_diff = diff
            optimal_c = c

    plt.figure(figsize=(10, 6))
    plt.plot(c_list, diff_list, label="NRC3",color='blue')
    plt.title('NRC3 Over K')
    plt.xlabel('K')
    plt.ylabel('NRC3')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.save_dir}KwithNRC3.png")
    plt.close()
    path = os.path.join(args.save_dir, 'gamma.json')
    data = {
        'gamma': c_list,
        'NRC3': diff_list
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, default=to_serializable)
        
    return optimal_c, min_diff

def find_K(W,Sigma_sqrt, args,device):
    min_diff = float('inf')
    optimal_K = 0
    K_list=[]
    diff_list=[]
    lambda_min = torch.min(torch.linalg.eigvalsh(Sigma_sqrt)).item() 
    print("lambda_min:",lambda_min)
    I = torch.eye(args.y_dim, device=device)
    
    for K in torch.linspace(0, math.sqrt(lambda_min)/math.sqrt(args.lambda_W+1e-10)*1.05, 1000):      
        diff = torch.norm((W @ W.T) - K*(Sigma_sqrt/math.sqrt(args.lambda_W+1e-10)-K * I), p='fro').item()
        K_list.append(K)
        diff_list.append(diff)
        if diff < min_diff:
            min_diff = diff
            optimal_K = K
    plt.figure(figsize=(10, 6))
    plt.plot(K_list, diff_list, label="NRC2",color='blue')

    plt.title('NRC2 Over K')
    plt.xlabel('K')
    plt.ylabel('NRC2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.save_dir}oldKwithNRC2.png")
    plt.close()
    return optimal_K, min_diff

def compute_pca(H, n_components):
    mean_H = torch.mean(H, dim=0)
    H_centered = H - mean_H
    covariance_matrix = H_centered.T @ H_centered / H.size(0)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix, UPLO='U')
    return eigenvectors[:, -n_components:]

def orthogonalize(W):
    Q, R = torch.linalg.qr(W)
    return Q

def compute_principal_angles(H_pca, W_orth):

    product = H_pca @ W_orth
    U, singular_values, V = torch.linalg.svd(product)
    angles = torch.acos(singular_values)
    return angles[0].item(),angles[1].item()


def gram_schmidt(W):
    U = torch.empty_like(W)
    U[0, :] = W[0, :] / torch.norm(W[0, :], p=2)

    proj = torch.dot(U[0, :], W[1, :]) * U[0, :]
    ortho_vector = W[1, :] - proj
    U[1, :] = ortho_vector / torch.norm(ortho_vector, p=2)

    return U

def nrc1N(H, n,device):

    mean_H = torch.mean(H, dim=0)
    H = H / (torch.norm(H, dim=1, keepdim=True) + 1e-8)
    H_np = H.cpu().numpy()
    pca_for_H = PCA(n_components=31)
    pca_for_H.fit(H_np)
    
    H_normalized = H / (torch.norm(H, dim=1, keepdim=True) + 1e-8)

    H_centered = H - mean_H
    H_pca = torch.tensor(pca_for_H.components_[:max(2, 6), :], device=device) 
    H_U = gram_schmidt(H_pca)
    P_H = H_U[:3, :].T @ H_U[:3, :]
    covariance_matrix = H_centered.T @ H_centered / H.size(0)

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix, UPLO='U')
    principal_components = eigenvectors[:, -n:]
    principal_components1 = eigenvectors[:, -1:]
    principal_components3 = eigenvectors[:, -3:]
    principal_components4 = eigenvectors[:, -4:]
    principal_components5 = eigenvectors[:, -5:]
    H_proj = H_normalized @ principal_components @ principal_components.T
    norm = torch.norm(H_normalized @ P_H - H_normalized).item() ** 2 / len(H)
    
    H_proj1 = H_centered @ principal_components1 @ principal_components1.T
    H_proj3 = H_centered @ principal_components3 @ principal_components3.T
    H_proj4 = H_centered @ principal_components4 @ principal_components4.T
    H_proj5 = H_centered @ principal_components5 @ principal_components5.T
    norm1 = torch.norm(H_centered - H_proj1, p='fro') ** 2 / H.size(0)
    norm3 = torch.norm(H_centered - H_proj3, p='fro') ** 2 / H.size(0)
    norm4 = torch.norm(H_centered - H_proj4, p='fro') ** 2 / H.size(0)
    norm5 = torch.norm(H_centered - H_proj5, p='fro') ** 2 / H.size(0)
    
    
    sorted_eigenvalues, _ = torch.sort(eigenvalues, descending=True)
    total_variance = torch.sum(sorted_eigenvalues).item()
    explained_variance_ratio = (sorted_eigenvalues / total_variance).tolist()

    
    return norm,norm1.item(),norm3.item(),norm4.item(),norm5.item(),explained_variance_ratio


def nrc1(H, n):

    mean_H = torch.mean(H, dim=0)
    H_centered = H
    covariance_matrix = H_centered.T @ H_centered / H.size(0)

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix, UPLO='U')
    principal_components = eigenvectors[:, -n:]
    principal_components1 = eigenvectors[:, -1:]
    principal_components3 = eigenvectors[:, -3:]
    principal_components4 = eigenvectors[:, -4:]
    principal_components5 = eigenvectors[:, -5:]
    H_proj = H_centered @ principal_components @ principal_components.T
    norm = torch.norm(H_centered - H_proj, p='fro') ** 2 / H.size(0) 
    
    H_proj1 = H_centered @ principal_components1 @ principal_components1.T
    H_proj3 = H_centered @ principal_components3 @ principal_components3.T
    H_proj4 = H_centered @ principal_components4 @ principal_components4.T
    H_proj5 = H_centered @ principal_components5 @ principal_components5.T
    norm1 = torch.norm(H_centered - H_proj1, p='fro') ** 2 / H.size(0)
    norm3 = torch.norm(H_centered - H_proj3, p='fro') ** 2 / H.size(0)
    norm4 = torch.norm(H_centered - H_proj4, p='fro') ** 2 / H.size(0)
    norm5 = torch.norm(H_centered - H_proj5, p='fro') ** 2 / H.size(0)
    
    
    sorted_eigenvalues, _ = torch.sort(eigenvalues, descending=True)
    total_variance = torch.sum(sorted_eigenvalues).item()
    explained_variance_ratio = (sorted_eigenvalues / total_variance).tolist()

    
    return norm.item(),norm1.item(),norm3.item(),norm4.item(),norm5.item(),explained_variance_ratio


def nrc2(H, W):

    U = gram_schmidt(W)
    P_E = torch.mm(U.T, U)
    H_proj = torch.mm(H, P_E)
    norm = torch.norm(H - H_proj, p='fro')** 2 / H.size(0)
    return norm.item()

def nrc2N(H, W):

    U = gram_schmidt(W)
    P_E = torch.mm(U.T, U)
    #W = W / (torch.norm(W, dim=1, keepdim=True) + 1e-8)
    #inverse_mat = torch.inverse(W @ W.T)
    #P_W = W.T @ inverse_mat @ W
    H_normalized = H / (torch.norm(H, dim=1, keepdim=True) + 1e-8)
    H_projected = torch.mm(H_normalized,P_E)
    norm = torch.norm(H_normalized-H_projected, p='fro') ** 2 / H.size(0)
    
    return norm.item()
# def nrc2N(H, W):

#     U = gram_schmidt(W)
#     P_E = torch.mm(U.T, U)
#     inverse_mat = torch.inverse(W @ W.T)
#     P_W = W.T @ inverse_mat @ W
#     H_normalized = H / (torch.norm(H, dim=1, keepdim=True) + 1e-8)
#     norm = torch.norm(H_normalized @ P_W - H_normalized).item() ** 2 / len(H)
    
#     return norm
def angle_between_vectors(v1, v2):
    unit_v1 = v1 / torch.norm(v1)
    unit_v2 = v2 / torch.norm(v2)
    cos_angle = torch.dot(unit_v1, unit_v2)
    angle = torch.acos(cos_angle)
    return angle.item()

def calculate_metrics(metrics, device,epoch, args,y_metrics):
    result = {}
    y = metrics['targets'].to(device)  #(B,2)
    Wh = metrics['outputs'].to(device) #(B,2)
    W = metrics['weights'].to(device) #(2,512)
    H = metrics['embeddings'].to(device) #(B,512)
    all_y = y_metrics['targets'].to(device) #(N,2)
    col1 = all_y[:, 0]
    col2 = all_y[:, 1]
    random_samples = np.random.choice(range(1, H.size(0)), size=100, replace=True)
    sampled_y = y[random_samples]
    sampled_H = H[random_samples]
    
    angles_y = []
    angles_H = []

    for i in range(100):
        for j in range(i + 1, 100):
            angle_y = angle_between_vectors(sampled_y[i], sampled_y[j])
            angle_H = angle_between_vectors(sampled_H[i], sampled_H[j])
            angles_y.append(angle_y)
            angles_H.append(angle_H)
    
    correlation = torch.corrcoef(torch.stack((col1, col2)))[0, 1]
    print('Correlation Coefficient:', correlation)
    print("Y",y.shape)
    print("Wh",Wh.shape)
    print("W",W.shape)
    print("W0",W[0].shape)
    print("H",H.shape)
    print("All_Y,",all_y.shape)
    H_norm = F.normalize(H, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    W_norm = F.normalize(W, p=2, dim=1)
    Wh_np_cpu = Wh.cpu().numpy()
    y_np_cpu = y.cpu().numpy()
    result["R2_score"]= r2_score(y_np_cpu, Wh_np_cpu)
    result['W_norm_square'] = (torch.norm(W, p=2).item())**2
    mean_values = all_y.mean(dim=0, keepdim=True)
    all_y_centered = all_y - mean_values
    all_y_norm = F.normalize(all_y,p=2, dim=1)
    WW = W @ W.T
    norm_WW = torch.norm(WW, p='fro')
    result['norm_H'] = torch.norm(H, p='fro').item()
    WW_norm = WW/norm_WW
    result['WW11_norm'] = WW_norm[0,0].item()
    result['WW12_norm'] = WW_norm[0,1].item()
    result['WW22_norm'] = WW_norm[1,1].item()
    Sigma = torch.matmul(all_y_centered.T, all_y_centered) / all_y_centered.size(0)
    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    Sigma_sqrt = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T

    W_norm_square_theory = args.lambda_H * (Sigma_sqrt / torch.sqrt(torch.tensor(args.lambda_H * (args.lambda_W+1e-10), device=device)) - torch.eye(args.y_dim, device=device))
    result['K'],result['NRC3'] = find_c(W, Sigma_sqrt, args,device)
    result['C'],result['NC2_K'] = find_K(W, Sigma_sqrt, args,device)
    del Sigma,eigenvalues, eigenvectors,sqrt_eigenvalues

    Sigma_sqrt_inverse = torch.linalg.inv(Sigma_sqrt)
    y_delta = torch.mm(y_norm ,Sigma_sqrt_inverse) 

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
    H_np = H.cpu().detach().numpy()
    pca_for_H = PCA(n_components=args.y_dim)
    #H_pca = pca_for_H.fit_transform(H_np) 
    #H_reconstruct = pca_for_H.inverse_transform(H_pca)
    result['projection_error_PCA'] = 1 #np.mean(np.square(H_np - H_reconstruct))
    

    #H_pca = compute_pca(H, n_components=args.y_dim)

    W_orth = orthogonalize(W)

    #angles0,angles1 = compute_principal_angles(H_pca, W_orth)

    result['H_W_angles0'], result['H_W_angles1']= 1,1 #angles0,angles1

    # Cosine similarity of Y and H post PCA
    #H_pca_norm = F.normalize(torch.tensor(H_pca).float().to(device), p=2, dim=1)
    #cos_sim_y_h_after_pca = torch.mm(H_pca_norm, y_norm.T)
    result['cos_sim_y_h_postPCA'] = 1 #cos_sim_y_h_after_pca.diag().mean().item()

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
    result['NRC1'],result['NRC1_1'],result['NRC1_3'],result['NRC1_4'],result['NRC1_5'],Explained_PCA_ratio = nrc1(H,args.y_dim)
    result['NRC1N'],_ ,_ ,_ ,_ ,_= nrc1N(H,args.y_dim,device)
    result["Explained_PCA_ratio1"]= Explained_PCA_ratio[0]
    result["Explained_PCA_ratio2"]= Explained_PCA_ratio[1]
    result["Explained_PCA_ratio3"]= Explained_PCA_ratio[2]
    result["Explained_PCA_ratio4"]= Explained_PCA_ratio[3]
    result["Explained_PCA_ratio5"]= Explained_PCA_ratio[4]
    result['NRC2'] = nrc2(H,W)
    result['NRC2N'] = nrc2N(H,W)


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
    scatter = plt.scatter(H_coordinates_norm[:, 0], H_coordinates_norm[:, 1], c=colors, cmap='viridis',vmin=vmin, vmax=vmax)
    plt.colorbar(scatter, label='y1/y2 ratio')
    plt.xlabel('W1')
    plt.ylabel('W2')
    plt.title('2D Coordinates Colored by y1/y2 Ratio')
    plt.grid(True)
    plt.show()
    plt.savefig(f"{args.save_dir}figs/{epoch}_y_fig1.png")

    plt.figure(figsize=(10, 6))
    plt.scatter(angles_H, angles_y, alpha=0.7)
    plt.title('Comparison of Angles between All Combinations')
    plt.xlabel('Angles between h vectors (degrees)')
    plt.ylabel('Angles between y vectors (degrees)')
    plt.grid(True)
    plt.show()
    plt.savefig(f"{args.save_dir}figs/{epoch}_h_y.png")

    return result

def plot_metrics_over_epochs(all_results, all_results_valid, epoch, save_dir):
    
    colors = ['#00008B', '#006400','#8B0000','#ADD8E6' ,'#90EE90', '#FFA07A']
    plt.figure(figsize=(10, 6))
    for i in range(0,5):
        plt.plot(range(1, epoch + 1), all_results[f'Explained_PCA_ratio{i+1}'], label=f"{i+1}", color=colors[i])
    plt.title('pca_n Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('ratio')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}pca_ratio.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['R2_score'], label="Train", color='blue')
    plt.plot(range(1, epoch + 1), all_results_valid['R2_score'], label="Test", color='red')
    plt.title('R2_score Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('R2_score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}R2_score.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['WW11_norm'], label="WW11", color='blue')
    plt.plot(range(1, epoch + 1), all_results['WW12_norm'], label="WW12", color='red')
    plt.plot(range(1, epoch + 1), all_results['WW22_norm'], label="WW22", color='green')
    plt.title('Check Covergence of WW')
    plt.xlabel('Epoch')
    plt.ylabel('WW')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}WW.png")
    plt.close()
 
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['NRC1'], label='NRC1', color='red')
    plt.plot(range(1, epoch + 1), all_results['NRC1_1'], label='NRC1_1', color='blue')
    plt.plot(range(1, epoch + 1), all_results['NRC1_3'], label='NRC1_3', color='yellow')
    plt.plot(range(1, epoch + 1), all_results['NRC1_4'], label='NRC1_4', color='green')
    plt.plot(range(1, epoch + 1), all_results['NRC1_5'], label='NRC1_5', color='cyan')
    plt.title('Train NRC1 Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Best NRC1')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}NRC1_5.png")
    plt.close()
    
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
    plt.plot(range(1, epoch + 1), all_results['H_W_angles0'], label='H_W_angles0', color='blue')
    plt.plot(range(1, epoch + 1), all_results['H_W_angles1'], label='H_W_angles1', color='red')
    plt.title('H_W_angles Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('H_W_angles')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}H_pca_W_angles.png")
    plt.close()


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['NRC3'], label='NRC3', color='blue')
    plt.title('Train NRC3 Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Best NRC3')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}NRC3.png")
    plt.close()


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['NRC2'], label='NRC2', color='blue')
    plt.title('Train NRC2 Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Best NRC2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}NRC2.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    #plt.plot(range(1, epoch + 1), all_results['NRC2'], label='NRC2', color='blue')
    plt.plot(range(1, epoch + 1), all_results['NRC2N'], label='NRC2N', color='red')
    plt.title('Train NRC2 and NRC2N Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Best NRC2 and NRC2N')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}NRC2N.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    #plt.plot(range(1, epoch + 1), all_results['NRC1'], label='NRC1', color='blue')
    plt.plot(range(1, epoch + 1), all_results['NRC1N'], label='NRC1N', color='red')
    plt.title('Train NRC1 and NRC1N Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Best NRC1 and NRC1N')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}NRC1N.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['C'], label='C', color='blue')
    plt.title('Train C Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Best C')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}C.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['NC2_K'], label='NC2_K', color='blue')
    plt.title('Train NC2_K Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Best NC2_K')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}NC2_K.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['K'], label='K', color='blue')
    plt.title('Train K Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Best K')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}K.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), all_results['norm_H'], label='H', color='blue')
    plt.title('norm_H Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('norm_H')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}norm_H.png")
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
    metrics_projection = ['projection_error_PCA', 'NRC1']
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
    plt.savefig(f"{save_dir}NRC1.png")
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
