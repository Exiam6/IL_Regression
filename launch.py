import os
import h5py
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from torch.nn.functional import mse_loss
from dataset import NumpyDataset, transform  
from model import RegressionResNet 

        
def main():

    save_dir = "/scratch/zz4330/IL_Regression_speed/Result/Case1_W5e-4"
    os.makedirs(save_dir, exist_ok=True)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug = False
    case2 = False
    y_dim = 1
    num_epochs = 500
    learning_rate = 1e-3
    lambda_H = 1e-5
    lambda_W = 5e-4
    sampling_rate = 0.1
    start = 61
    train_dataset = NumpyDataset('/scratch/zz4330/Carla/Train/images.npy', '/scratch/zz4330/Carla/Train/targets.npy',transform=transform)
    val_dataset = NumpyDataset('/scratch/zz4330/Carla/Val/images.npy', '/scratch/zz4330/Carla/Val/targets.npy', transform=transform)
    train_data_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_data_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4)
    
    model = RegressionResNet(pretrained=True, num_outputs=1).to(device)
    checkpoint_path = f'/scratch/zz4330/IL_Regression_speed/Result/Case1_W5e-4/checkpoints/model_checkpoint_epoch_60.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    for name, param in model.named_parameters():
        print(f"{name}: {param.size()}")
    criterion = nn.MSELoss()
   
    if case2:
        param_groups = [
        {'params': model.model.fc.parameters(), 'weight_decay': lambda_W},  # Last layer
        {'params': (p for n, p in model.named_parameters() if n not in ['model.fc.weight', 'model.fc.bias']), 'weight_decay': 0}  
        ]
        optimizer = torch.optim.SGD(param_groups, lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lambda_W)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch_correlations = []
    training_losses = []
    validation_losses = []
    projection_errors = []
    projection_errors_valid = []
    projection_errors_random = []
    projection_errors_W = []
    projection_errors_inputs = []
    cos_sim_y_Wh = []
    cos_sim_W = []
    cos_sim_H = []
    cos_sim_y_h_postPCA = []
    cos_sim_y_h_beforePCA = []
    MSE_cos = []
    projection_error_h2W_list = []
    projection_error_h2W_list_valid = []
    projection_error_h2W_E_list = []
    projection_error_h2W_E_list_valid = []
    if checkpoint_path:
        epoch_correlations = checkpoint['epoch_correlations']
        training_losses = checkpoint['training_losses']
        validation_losses = checkpoint['validation_losses']
        projection_errors = checkpoint['projection_errors']
        projection_errors_valid = checkpoint['projection_errors_valid']
        projection_errors_random = checkpoint['projection_errors_random']
        projection_errors_W = checkpoint['projection_errors_W']
        projection_errors_inputs = checkpoint['projection_errors_inputs']
        cos_sim_y_Wh = checkpoint['cos_sim_y_Wh']
        cos_sim_W = checkpoint['cos_sim_W']
        cos_sim_H = checkpoint['cos_sim_H']
        cos_sim_y_h_postPCA = checkpoint['cos_sim_y_h_postPCA']
        cos_sim_y_h_beforePCA = checkpoint['cos_sim_y_h_beforePCA']
        MSE_cos = checkpoint['MSE_cos']
        projection_error_h2W_list = checkpoint['projection_error_h2W_list']
        projection_error_h2W_list_valid = checkpoint['projection_error_h2W_list_valid']
        projection_error_h2W_E_list = checkpoint['projection_error_h2W_E_list']
        projection_error_h2W_E_list_valid = checkpoint['projection_error_h2W_E_list_valid']
    for epoch in range(start,num_epochs+1):
        embeddings_list = []
        embeddings_list_valid = []
        random_embeddings_list = []
        W_list = []
        weight_list = []
        targets_list =[]
        output_list = []
        targets_list_valid =[]
        output_list_valid = []
        input_list=[]
        
        model.train()
        running_loss = 0.0
        train_count=0
        train_limit = 200  # limit samples plotted in PCA due to out of memory. The samples here are randomly selected

        for batch in tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            
            if debug:
                if train_count>10:
                    break
            images = batch['image'].to(device)
            targets = batch['targets'].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            embeddings = model.get_last_layer_embeddings(images)
            last_layer_weights = model.model.fc.weight
            #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #with record_function("model_training"):
            if train_count <= train_limit and random.random() < sampling_rate:
                train_count+=1

                embeddings_list.append(embeddings.cpu().numpy())
                random_embeddings_list.append(np.random.normal(0,1, embeddings.shape))
                targets_list.append(targets.cpu().numpy())
                output_list.append(outputs.detach().cpu().numpy())
                input_list.append(images.cpu().numpy())
                weight_list=[model.model.fc.weight.detach().cpu().numpy()]

            loss = criterion(outputs, targets)
            l2reg_H = lambda_H * torch.norm(embeddings, 2)
            #l2reg_W = lambda_W * torch.norm(last_layer_weights, 2)
            if case2:
                loss = loss + l2reg_H

            loss.backward()
            optimizer.step()
       

            running_loss += loss.item()
            

        avg_train_loss = running_loss / len(train_data_loader)
        training_losses.append(avg_train_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_data_loader)}")
        
        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            valid_count = 0
            processed_samples = 0
            
            for batch in tqdm(val_data_loader, desc=f"Validation"): 
                if debug:
                    if valid_count>10:
                        break
                images = batch['image'].to(device)
                targets = batch['targets'].to(device)
                
                outputs = model(images)
            
                embeddings_valid = model.get_last_layer_embeddings(images)
                if len(embeddings_list_valid) <100 and (random.random() < 0.3 or len(embeddings_list_valid)<30):
                    embeddings_list_valid.append(embeddings_valid.cpu().numpy())
                    targets_list_valid.append(targets.cpu().numpy())
                
                loss = criterion(outputs, targets)
                
                total_val_loss += loss.item() 
                valid_count +=1

        avg_val_loss = total_val_loss / len(val_data_loader)
        validation_losses.append(avg_val_loss)
        print(f"Validation Loss: {total_val_loss /len(val_data_loader)}")

        # Preprocess
        y = torch.tensor(targets_list)  #(11,218,3)
        Wh = torch.tensor(output_list)  #(11,218,3)
        weight_matrix = np.concatenate(weight_list,axis=0) #(33, 512)
        print(Wh.shape)
        print(f"Weight matrix dim: {weight_matrix.shape}")

        
        # PCA for Wi
        pca_weights = PCA(n_components=y_dim) 
        reduced_weights = pca_weights.fit_transform(weight_matrix)
        reconstructed_weights = pca_weights.inverse_transform(reduced_weights)

        weight_projection_error = np.mean(np.square(weight_matrix - reconstructed_weights))
        print(f"Weight matrix projection error: {weight_projection_error}")
        projection_errors_W.append(weight_projection_error)
        
        # PCA for hi
        all_embeddings = np.concatenate(embeddings_list, axis=0) #PCA on Train
        all_targets = np.concatenate(targets_list, axis=0)
    
        print("Shape of all_embeddings before PCA:", all_embeddings.shape)
        all_embeddings_reshaped = all_embeddings.reshape(all_embeddings.shape[0], -1)

        print("Shape of all_targets before PCA:", all_targets.shape)

        all_embeddings_norm = F.normalize(torch.tensor(all_embeddings_reshaped).float(), p=2, dim=1)
        all_targets_norm = F.normalize(torch.tensor(all_targets).float(), p=2, dim=1)


        # PCA for hi
        pca_H = PCA(n_components=y_dim)
        reduced_embeddings = pca_H.fit_transform(all_embeddings_norm) 
        reconstructed_embeddings = pca_H.inverse_transform(reduced_embeddings)
        projection_error = np.mean(np.square(all_embeddings_norm.numpy() - reconstructed_embeddings))
        
        print(f"Projection error: {projection_error}") # projection_error for 3D
        projection_errors.append(projection_error)

        #PCA for random hi
        all_random_embeddings = np.concatenate(random_embeddings_list, axis=0)
        all_random_embeddings_reshaped = all_random_embeddings.reshape(all_random_embeddings.shape[0], -1)
        all_random_embeddings_norm = F.normalize(torch.tensor(all_random_embeddings_reshaped).float(), p=2, dim=1)

        pca_random = PCA(n_components=y_dim)
        reduced_embeddings_random = pca_random.fit_transform(all_random_embeddings_norm)
        reconstructed_embeddings_random = pca_random.inverse_transform(reduced_embeddings_random)
        projection_error_random = np.mean(np.square(all_random_embeddings_norm.numpy() - reconstructed_embeddings_random))
        print(f"Projection error for Random: {projection_error_random}") # projection_error for 3D
        projection_errors_random.append(projection_error_random)
        
        # PCA for hi
        all_embeddings_valid = np.concatenate(embeddings_list_valid, axis=0) #PCA on Train
        all_targets_valid = np.concatenate(targets_list_valid, axis=0)
    
        print("Shape of all_embeddings before PCA:", all_embeddings_valid.shape)
        all_embeddings_reshaped_valid = all_embeddings_valid.reshape(all_embeddings_valid.shape[0], -1)

        print("Shape of all_targets before PCA:", all_targets_valid.shape)

        all_embeddings_norm_valid = F.normalize(torch.tensor(all_embeddings_reshaped_valid).float(), p=2, dim=1)
        all_targets_norm_valid = F.normalize(torch.tensor(all_targets_valid).float(), p=2, dim=1)


        # PCA for hi
        pca_H_valid = PCA(n_components=y_dim)
        reduced_embeddings_valid = pca_H_valid.fit_transform(all_embeddings_norm_valid) 
        reconstructed_embeddings_valid = pca_H_valid.inverse_transform(reduced_embeddings_valid)
        projection_error_valid = np.mean(np.square(all_embeddings_norm_valid.numpy() - reconstructed_embeddings_valid))
        
        print(f"Projection error_valid: {projection_error_valid}") # projection_error for 3D
        projection_errors_valid.append(projection_error_valid)

        # PCA for input images
        input_images_flattened = np.concatenate(input_list, axis=0)
        input_images_flattened = input_images_flattened.reshape(input_images_flattened.shape[0], -1) 
        print(input_images_flattened.shape)
       
        
        pca_inputs = PCA(n_components=y_dim)
        reduced_inputs = pca_inputs.fit_transform(input_images_flattened)
        print(f"Shape of reduced_inputs: {reduced_inputs.shape}")
        
        reconstructed_inputs = pca_inputs.inverse_transform(reduced_inputs)
        print(f"Shape of reconstructed_inputs: {reconstructed_inputs.shape}")
       
        # Projection error for input images
        projection_error_inputs = np.mean(np.square(input_images_flattened - reconstructed_inputs))
        projection_errors_inputs.append(projection_error_inputs)
        
        print(f"Projection error for input images at epoch {epoch}: {projection_error_inputs}")

        # Three Cosine similarities

        #  between y and Wh
        print(y.shape,Wh.shape)
        cos_sim_epoch = F.cosine_similarity(y, Wh, dim=1)
        cos_sim_y_Wh.append(cos_sim_epoch.mean().item())
        print(f"Cosine similarity between y and Wh is : {cos_sim_y_Wh[-1]}")

        # between Wi's
        cos_sim_matrix = cosine_similarity(weight_matrix)
        np.fill_diagonal(cos_sim_matrix, np.nan)
        average_cos_sim = np.nanmean(cos_sim_matrix)
        cos_sim_W.append(average_cos_sim)
        print(f"Cosine similarity between rows of W: {average_cos_sim}")
    

        # between hi's
        cos_sim_matrix_H = cosine_similarity(all_embeddings_norm)
        np.fill_diagonal(cos_sim_matrix_H, np.nan)
        average_cos_sim_H = np.nanmean(cos_sim_matrix_H)
        cos_sim_H.append(average_cos_sim_H)
        print(f"Cosine similarity between rows of H: {average_cos_sim_H}")
        # between yi and hi post PCA
        all_embeddings_after_pca_norm = F.normalize(torch.tensor(reduced_embeddings).float(), p=2, dim=1)
        cos_sim_after_pca = torch.mm(all_embeddings_after_pca_norm, all_targets_norm.transpose(0, 1))
        cos_sim_y_h_postPCA_mean = torch.mean(torch.diag(cos_sim_after_pca))
        cos_sim_y_h_postPCA.append(cos_sim_y_h_postPCA_mean)
        print(f"Cosine similarity between yi and hi post PCA: {cos_sim_y_h_postPCA_mean}")


        #  MSE between 1) Cosine Similarity between  h_i and h_j 1) Cosine Similarity between  y_i and y_j

        cos_sim_embeddings = np.dot(all_embeddings_norm.numpy(), all_embeddings_norm.numpy().T)
        cos_sim_targets = np.dot(all_targets_norm.numpy(), all_targets_norm.numpy().T)
        cos_sim_embeddings_tensor = torch.tensor(cos_sim_embeddings, dtype=torch.float32)
        cos_sim_targets_tensor = torch.tensor(cos_sim_targets, dtype=torch.float32)
        
        # Get the upper triangular indices excluding the diagonal
        n = cos_sim_embeddings_tensor.size(0)  
        indices = torch.triu_indices(n, n, offset=1)  
        upper_tri_embeddings = cos_sim_embeddings_tensor[indices[0], indices[1]]
        upper_tri_targets = cos_sim_targets_tensor[indices[0], indices[1]]
        mse_value = mse_loss(upper_tri_embeddings, upper_tri_targets)
        MSE_cos.append(mse_value)

        print("MSE between cosine similarities of embeddings and targets:", mse_value.item())
        # Correlation between distance of hi and yi

        # Calculate pairwise distances among reduced embeddings
        embeddings_distances = squareform(pdist(reduced_embeddings, 'euclidean'))
        print(f"Shape of all_targets:{all_targets.shape}")
        # Calculate pairwise distances among targets
        targets_distances = squareform(pdist(all_targets, 'euclidean'))

        correlation_matrix = np.corrcoef(embeddings_distances.flatten(), targets_distances.flatten())
        correlation = correlation_matrix[0, 1]
        print("Correlation between distances in embeddings and target values:", correlation)
        epoch_correlations.append(correlation)
        corrected_targets = all_targets[:, 0]  # all_targets was (374, 3)
        
        def gram_schmidt(W):
            U = torch.empty_like(W)
            U[0, :] = W[0, :] / torch.norm(W[0, :], p=2)  
            return U

        # Projection of h onto W
        W = torch.tensor(weight_matrix, dtype=torch.float32)
        h = torch.tensor(all_embeddings, dtype=torch.float32)
        W_norm = W.T / torch.norm(W.T, dim=1, keepdim=True)
        P = torch.mm(W.T, torch.mm(torch.inverse(torch.mm(W, W.T)), W))
        h_projected = torch.mm(h, P)
        projection_error_h2W = mse_loss(h_projected,h).item()

        projection_error_h2W_list.append(projection_error_h2W)
        print("Projection error from h to W:", projection_error_h2W)

        h_valid = torch.tensor(all_embeddings_valid).float()
        h_projected_valid = torch.mm(h_valid, P)
        projection_error_h2W_valid = mse_loss(h_projected_valid,h_valid).item()
        projection_error_h2W_list_valid.append(projection_error_h2W_valid)
        print("Projection error from h to W for valid:", projection_error_h2W_valid)

        U = gram_schmidt(W)
        P_E = torch.mm(U.T, U)  # Projection matrix using orthonormal basis
        h_projected_E = torch.mm(h, P_E)
        projection_error_h2W_E = mse_loss(h_projected_E, h).item()
        projection_error_h2W_E_list.append(projection_error_h2W_E)
        print("Projection error from h to W with E for Train:", projection_error_h2W_E)

        h_projected_E_valid = torch.mm(h_valid, P_E)
        projection_error_h2W_E_valid = mse_loss(h_projected_E_valid,h_valid).item()
        projection_error_h2W_E_list_valid.append(projection_error_h2W_E_valid)
        print("Projection error from h to W with E for valid:", projection_error_h2W_E_valid)
        h_coordinates = torch.mm(h, U.T)
        print("Coordinates of h in the basis spanned by W:", h_coordinates)
        h_coordinates_np = h_coordinates.numpy()
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch + 1), cos_sim_y_Wh, label='Cosine Similarity between $y_i$ and $Wh_i$')
        plt.plot(range(1, epoch + 1), cos_sim_W, label='Cosine Similarity between $W_i$')
        plt.plot(range(1, epoch + 1), cos_sim_H, label='Cosine Similarity between $H_i$')
        plt.plot(range(1, epoch + 1), cos_sim_y_h_postPCA, label='Cosine Similarity between $y_i$ and $h_i$')
        plt.xlabel('Epoch')
        plt.ylabel('Cosine Similarity')
        plt.title('Cosine Similarity Trends Over Epochs')
        plt.legend()
        plt.savefig(os.path.join(save_dir, "Angle_across_epochs.png"))
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(range(1,epoch + 1), epoch_correlations, marker='o', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Correlation between Embeddings and Target Distances')
        plt.title('Correlation across Epochs')

        plt.xticks(range(1, num_epochs + 1)) 

        plt.savefig(os.path.join(save_dir, f"Correlation.png"))
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch + 1), training_losses, label='Training Loss', marker='o', linestyle='-')
        plt.plot(range(1, epoch + 1), validation_losses, label='Validation Loss', marker='o', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Across Epochs')
        plt.legend()
        plt.savefig(os.path.join(save_dir, "training_validation_loss.png"))

        plt.figure(figsize=(10, 6))
        plt.plot(range(1,epoch + 1), MSE_cos, marker='o', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Error')
        plt.title('MSE for cos between y and h')
        plt.savefig(os.path.join(save_dir, f"MSE for cos between y and h.png"))
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch + 1), projection_errors, label = 'H',marker='o', linestyle='-', color='blue')
        plt.plot(range(1, epoch + 1), projection_errors_valid, label = 'H Valid',marker='o', linestyle='-', color='red')
        plt.plot(range(1, epoch + 1), projection_errors_random, label = 'Random_H', marker='o', linestyle='-', color='green')
        plt.title('PCA Projection Error')
        plt.xlabel('Epoch')
        plt.ylabel('PCA Projection Error')
        plt.legend()
        plt.savefig(os.path.join(save_dir, "PCA_Projection_Error_For.png"))

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch + 1), projection_error_h2W_E_list, label = 'H to W Train',marker='o', linestyle='-', color='blue')
        plt.plot(range(1, epoch + 1), projection_error_h2W_E_list_valid, label = 'H to W Valid',marker='o', linestyle='-', color='red')
        plt.title('H2W Projection Error')
        plt.xlabel('Epoch')
        plt.ylabel('H2W Projection Error')
        plt.legend()
        plt.savefig(os.path.join(save_dir, "H2W_Projection_Error.png"))

        if epoch%10==0:
            first_label = all_targets[:, 0]
            scaler = MinMaxScaler()
            y = np.zeros(h_coordinates_np.shape[0])
            colors = scaler.fit_transform(first_label.reshape(-1, 1)).flatten()
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(h_coordinates_np[:, 0], y,c=colors,  cmap='plasma', edgecolor='k', alpha=0.7, s=50)
            plt.colorbar(scatter)
            plt.xlabel('Component 1')

            plt.title('Projection of h onto the Space Spanned by W')
            plt.savefig(os.path.join(save_dir, f"First Label H2W for Train at epoch{epoch}.png"))

            y = np.zeros(reduced_embeddings.shape[0])
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(reduced_embeddings[:, 0],y, c=colors,  cmap='plasma', edgecolor='k', alpha=0.7, s=50)
            plt.colorbar(scatter)
            plt.xlabel('Principal Component 1')
            plt.title('PCA Projection for Train by First Label')
            plt.savefig(os.path.join(save_dir, f"First Label PCA for Train at epoch{epoch}.png"))

        
            first_label_valid = all_targets_valid[:, 0]
            scaler = MinMaxScaler()
            colors_valid = scaler.fit_transform(first_label_valid.reshape(-1, 1)).flatten()
            y = np.zeros(reduced_embeddings_valid.shape[0])
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(reduced_embeddings_valid[:, 0],y, c=colors_valid, cmap='plasma', edgecolor='k', alpha=0.7, s=50)
            plt.colorbar(scatter)
            plt.xlabel('Principal Component 1')
            plt.title('PCA Projection for Val by First Label')
            plt.savefig(os.path.join(save_dir, f"First Label PCA for Val at epoch{epoch}.png"))


            checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': training_losses[-1],  # Most recent training loss
            'correlation': epoch_correlations[-1],  # Most recent epoch correlation
            'projection_errors_w': projection_errors_W[-1],
            'projection_errors_Input': projection_errors_inputs[-1],
            'training_losses': training_losses,  # Full list
            'validation_losses': validation_losses,
            'epoch_correlations': epoch_correlations,
            'projection_errors': projection_errors,
            'projection_errors_valid': projection_errors_valid,
            'projection_errors_random': projection_errors_random,
            'projection_errors_W': projection_errors_W,
            'projection_errors_inputs': projection_errors_inputs,
            'cos_sim_y_Wh': cos_sim_y_Wh,
            'cos_sim_W': cos_sim_W,
            'cos_sim_H': cos_sim_H,
            'cos_sim_y_h_postPCA': cos_sim_y_h_postPCA,
            'cos_sim_y_h_beforePCA': cos_sim_y_h_beforePCA,
            'MSE_cos': MSE_cos,
            'projection_error_h2W_list': projection_error_h2W_list,
            'projection_error_h2W_list_valid': projection_error_h2W_list_valid,
            'projection_error_h2W_E_list': projection_error_h2W_E_list,
            'projection_error_h2W_E_list_valid': projection_error_h2W_E_list_valid
            }

            checkpoint_filename = f'model_checkpoint_epoch_{epoch}.pth'
            save_dir_checkpoint = f'{save_dir}/checkpoints'
            os.makedirs(save_dir_checkpoint, exist_ok=True)
            checkpoint_path = os.path.join(save_dir_checkpoint, checkpoint_filename)

            torch.save(checkpoint, checkpoint_path)
            print(f'Model checkpoint saved at {checkpoint_path}')

            #checkpoint = torch.load(checkpoint_path)
            #model.load_state_dict(checkpoint['model_state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #epoch = checkpoint['epoch']
            #loss = checkpoint['loss']  


   
if __name__ == '__main__':
    main()

'''
for name, parameter in model.named_parameters():
        print(name, parameter)
        if is_in_layer(name, ['model.fc']):  
            param_groups[2]['params'].append(parameter)
        elif is_in_layer(name, ['model.layer4']):  
            param_groups[1]['params'].append(parameter)
        else:
            param_groups[0]['params'].append(parameter)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(reduced_inputs[:, 0], reduced_inputs[:, 1], reduced_inputs[:, 2], s=25, alpha=0.6)
            ax.set_title('3D PCA-reduced Input Images')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
            plt.savefig(os.path.join(save_dir, f"3D_PCA_Inputs_at_epoch{epoch}.png"))
            plt.show()

          
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_embeddings_3D[:, 0], reduced_embeddings_3D[:, 1], reduced_embeddings_3D[:, 2], c=corrected_targets.flatten(), cmap='tab10', s=25, alpha=0.6)
            
            ax.set_title('PCA-reduced Embeddings Colored by Digit Label in 3D')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
            legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
            ax.add_artist(legend1)
            
            plt.savefig(os.path.join(save_dir, f"3D_PCA_H_at_epoch{epoch}.png"))

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(
                reduced_weights[:, 0], 
                reduced_weights[:, 1], 
                reduced_weights[:, 2], 
                cmap='tab10', 
                s=25, 
                alpha=0.6
            )

            ax.set_title('PCA-reduced Weight Matrix in 3D')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
            plt.savefig(os.path.join(save_dir, f"3D_PCA_W_at_epoch{epoch}.png"))
     
            plt.show()

'''
