"""
Centralized training module
Handles standard training of all three models
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim

from config import *


def train_classification_model(model, train_loader, test_loader, model_name, device, 
                               epochs=CENTRALIZED_EPOCHS, learning_rate=LEARNING_RATE):
    """Train a classification model (CNN or DenseNN)"""
    
    print(f"\n{'='*70}")
    print(f">> Training {model_name}")
    print(f"{'='*70}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_times': []
    }
    
    best_test_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = 100.0 * test_correct / test_total
        
        scheduler.step(test_loss)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
        
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(epoch_time)
        
        print(f"Epoch [{epoch+1:2d}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")
    
    model.load_state_dict(best_model_state)
    
    print(f"\n[OK] Training complete for {model_name}")
    print(f"   Best test accuracy: {best_test_acc:.2f}%")
    print(f"   Total training time: {sum(history['epoch_times']):.2f}s")
    
    return history


def train_autoencoder(model, train_loader, test_loader, device, 
                     epochs=CENTRALIZED_EPOCHS, learning_rate=LEARNING_RATE, 
                     alpha=AUTOENCODER_ALPHA):
    """Train autoencoder with hybrid loss (reconstruction + classification)"""
    
    print(f"\n{'='*70}")
    print(f">> Training AutoEncoder_IDS")
    print(f"{'='*70}")
    
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'recon_loss': [],
        'class_loss': [],
        'epoch_times': []
    }
    
    best_test_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_class_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            reconstruction, code, classification = model(batch_x)
            
            loss_recon = criterion_recon(reconstruction, batch_x)
            loss_class = criterion_class(classification, batch_y)
            loss = alpha * loss_recon + (1 - alpha) * loss_class
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += loss_recon.item()
            train_class_loss += loss_class.item()
            
            _, predicted = torch.max(classification.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_class_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                reconstruction, code, classification = model(batch_x)
                loss_recon = criterion_recon(reconstruction, batch_x)
                loss_class = criterion_class(classification, batch_y)
                loss = alpha * loss_recon + (1 - alpha) * loss_class
                
                test_loss += loss.item()
                _, predicted = torch.max(classification.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = 100.0 * test_correct / test_total
        
        scheduler.step(test_loss)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
        
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['recon_loss'].append(train_recon_loss)
        history['class_loss'].append(train_class_loss)
        history['epoch_times'].append(epoch_time)
        
        print(f"Epoch [{epoch+1:2d}/{epochs}] | "
              f"Total Loss: {train_loss:.4f} | Recon: {train_recon_loss:.4f} | Class: {train_class_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")
    
    model.load_state_dict(best_model_state)
    
    print(f"\n[OK] Training complete for AutoEncoder_IDS")
    print(f"   Best test accuracy: {best_test_acc:.2f}%")
    print(f"   Total training time: {sum(history['epoch_times']):.2f}s")
    
    return history

