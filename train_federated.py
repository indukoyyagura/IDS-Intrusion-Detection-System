"""
Federated learning training module
Implements FedAvg algorithm for distributed training
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim

from config import *


def train_local_model_classification(model, train_data, device, client_id, args):
    """Train classification model locally on one client's data"""
    
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
    
    epoch_loss = []
    for epoch in range(args['epochs']):
        batch_loss = []
        for batch_idx, (x, y) in enumerate(train_data):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            outputs = model(x)
            loss = criterion(outputs, y)
            
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
        
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        print(f"      Client {client_id} - Epoch {epoch+1}/{args['epochs']}, Loss: {epoch_loss[-1]:.6f}")
    
    final_loss = sum(epoch_loss) / len(epoch_loss)
    return model.state_dict(), final_loss


def train_local_model_autoencoder(model, train_data, device, client_id, args, alpha=AUTOENCODER_ALPHA):
    """Train autoencoder locally on one client's data"""
    
    model.to(device)
    model.train()
    
    criterion_recon = nn.MSELoss().to(device)
    criterion_class = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
    
    epoch_loss = []
    for epoch in range(args['epochs']):
        batch_loss = []
        for batch_idx, (x, y) in enumerate(train_data):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            reconstruction, code, classification = model(x)
            
            loss_recon = criterion_recon(reconstruction, x)
            loss_class = criterion_class(classification, y)
            loss = alpha * loss_recon + (1 - alpha) * loss_class
            
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
        
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        print(f"      Client {client_id} - Epoch {epoch+1}/{args['epochs']}, Loss: {epoch_loss[-1]:.6f}")
    
    final_loss = sum(epoch_loss) / len(epoch_loss)
    return model.state_dict(), final_loss


def federated_averaging(local_models_info):
    """Perform FedAvg aggregation (weighted averaging by sample count)"""
    
    total_samples = sum(info[2] for info in local_models_info)
    
    aggregated_params = None
    
    for state_dict, loss, num_samples in local_models_info:
        weight = num_samples / total_samples
        
        if aggregated_params is None:
            aggregated_params = {}
            for key, param in state_dict.items():
                aggregated_params[key] = param.clone() * weight
        else:
            for key, param in state_dict.items():
                aggregated_params[key] += param * weight
    
    return aggregated_params


def train_federated_model(model_class, model_name, input_dim, train_data_local_dict, 
                         train_data_local_num_dict, test_loader, device, 
                         is_autoencoder=False):
    """Train a model using federated learning"""
    
    print(f"\n{'='*70}")
    print(f">> Federated Training: {model_name}")
    print(f"{'='*70}")
    
    # Initialize model
    if is_autoencoder:
        model = model_class(input_dim, bottleneck_dim=AUTOENCODER_BOTTLENECK_DIM).to(device)
    else:
        model = model_class(input_dim).to(device)
    
    args_fed = {
        'epochs': LOCAL_EPOCHS,
        'learning_rate': FED_LEARNING_RATE,
        'comm_round': COMM_ROUND
    }
    
    history = {
        'global_losses': [],
        'client_losses': {i: [] for i in range(CLIENT_NUM_IN_TOTAL)},
        'round_times': [],
        'test_accuracy': []
    }
    
    for round_idx in range(COMM_ROUND):
        print(f"\nCommunication Round {round_idx + 1}/{COMM_ROUND}")
        print("-" * 60)
        
        round_start_time = time.time()
        local_models_info = []
        
        for client_idx in range(CLIENT_NUM_PER_ROUND):
            print(f"  Training on Client {client_idx}...")
            
            # Create local copy of global model
            if is_autoencoder:
                local_model = model_class(input_dim, bottleneck_dim=AUTOENCODER_BOTTLENECK_DIM)
            else:
                local_model = model_class(input_dim)
            local_model.load_state_dict(model.state_dict())
            
            train_data = train_data_local_dict[client_idx]
            num_samples = train_data_local_num_dict[client_idx]
            
            # Local training
            if is_autoencoder:
                local_state_dict, local_loss = train_local_model_autoencoder(
                    local_model, train_data, device, client_idx, args_fed
                )
            else:
                local_state_dict, local_loss = train_local_model_classification(
                    local_model, train_data, device, client_idx, args_fed
                )
            
            local_models_info.append((local_state_dict, local_loss, num_samples))
            history['client_losses'][client_idx].append(local_loss)
        
        print(f"  Performing FedAvg aggregation...")
        aggregated_params = federated_averaging(local_models_info)
        
        model.load_state_dict(aggregated_params)
        
        global_loss = sum(info[1] for info in local_models_info) / len(local_models_info)
        history['global_losses'].append(global_loss)
        
        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                if is_autoencoder:
                    _, _, classification = model(batch_x)
                    outputs = classification
                else:
                    outputs = model(batch_x)
                
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
        
        test_acc = 100.0 * test_correct / test_total
        history['test_accuracy'].append(test_acc)
        
        round_time = time.time() - round_start_time
        history['round_times'].append(round_time)
        
        print(f"  Round {round_idx + 1} completed in {round_time:.2f}s")
        print(f"  Global loss: {global_loss:.6f} | Test Accuracy: {test_acc:.2f}%")
    
    print(f"\nFederated Training Completed ({model_name})!")
    print(f"   Final global loss: {history['global_losses'][-1]:.6f}")
    print(f"   Final test accuracy: {history['test_accuracy'][-1]:.2f}%")
    print(f"   Total training time: {sum(history['round_times']):.2f}s")
    
    return model, history

