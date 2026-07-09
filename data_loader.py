"""
Data loading and preprocessing module
Handles CIC-IDS 2017 dataset loading, cleaning, and preparation
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib

from config import *


def load_and_combine_datasets():
    """Load and combine multiple CIC-IDS 2017 dataset files"""
    print("[LOAD] Loading datasets...")
    print("Loading multiple days to get a balanced mix of benign and attack traffic\n")
    
    dataframes = []
    
    # Load Monday (mostly benign)
    df_monday = pd.read_csv(os.path.join(DATA_DIR, DATA_FILES['Monday']))
    print(f"[+] Monday: {len(df_monday):,} samples")
    dataframes.append(df_monday.sample(n=min(50000, len(df_monday)), random_state=RANDOM_STATE))
    
    # Load Tuesday (various attacks)
    df_tuesday = pd.read_csv(os.path.join(DATA_DIR, DATA_FILES['Tuesday']))
    print(f"[+] Tuesday: {len(df_tuesday):,} samples")
    dataframes.append(df_tuesday.sample(n=min(50000, len(df_tuesday)), random_state=RANDOM_STATE))
    
    # Load Friday DDoS (DDoS attacks)
    df_friday_ddos = pd.read_csv(os.path.join(DATA_DIR, DATA_FILES['Friday_Afternoon_DDoS']))
    print(f"[+] Friday DDoS: {len(df_friday_ddos):,} samples")
    dataframes.append(df_friday_ddos.sample(n=min(50000, len(df_friday_ddos)), random_state=RANDOM_STATE))
    
    # Load Friday PortScan (port scanning attacks)
    df_friday_portscan = pd.read_csv(os.path.join(DATA_DIR, DATA_FILES['Friday_Afternoon_PortScan']))
    print(f"[+] Friday PortScan: {len(df_friday_portscan):,} samples")
    dataframes.append(df_friday_portscan.sample(n=min(50000, len(df_friday_portscan)), random_state=RANDOM_STATE))
    
    # Combine all dataframes
    df = pd.concat(dataframes, ignore_index=True)
    
    print(f"\n[DATA] Combined dataset: {len(df):,} samples")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    print(f"   Number of features: {len(df.columns)}")
    
    return df


def preprocess_data(df):
    """Complete data preprocessing pipeline"""
    print("\n[CONFIG] Data Preprocessing Pipeline")
    print("="*70)
    
    # Determine label column name
    label_col = ' Label' if ' Label' in df.columns else 'Label'
    
    # Step 1: Create binary classification
    print("\n1. Creating binary labels (BENIGN vs ATTACK)...")
    df['Binary_Label'] = df[label_col].apply(lambda x: 0 if x == 'BENIGN' else 1)
    print(f"   [+] Binary label distribution:")
    print(f"     BENIGN (0): {sum(df['Binary_Label']==0):,} samples")
    print(f"     ATTACK (1): {sum(df['Binary_Label']==1):,} samples")
    
    # Step 2: Handle infinite values
    print("\n2. Handling infinite values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'Binary_Label':
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"   [+] Replaced inf values with NaN")
    
    # Step 3: Handle missing values
    print("\n3. Handling missing values...")
    df.fillna(0, inplace=True)
    print(f"   [+] Filled NaN values with 0")
    
    # Step 4: Remove low-variance features
    print("\n4. Removing low-variance features...")
    feature_cols = [col for col in df.columns if col not in [label_col, 'Binary_Label']]
    original_feature_count = len(feature_cols)
    
    selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    X_temp = df[feature_cols].values
    selector.fit(X_temp)
    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]
    print(f"   [+] Kept {len(selected_features)}/{original_feature_count} features")
    
    # Step 5: Create feature matrix
    print("\n5. Creating feature matrix...")
    X = df[selected_features].values
    y = df['Binary_Label'].values
    print(f"   [+] Feature matrix shape: {X.shape}")
    print(f"   [+] Label vector shape: {y.shape}")
    print(f"   [+] Attack ratio: {np.mean(y):.2%}")
    
    # Step 6: Balance dataset
    print("\n6. Balancing dataset...")
    X_benign = X[y == 0]
    y_benign = y[y == 0]
    X_attack = X[y == 1]
    y_attack = y[y == 1]
    
    n_samples = min(len(X_benign), len(X_attack), BALANCE_SAMPLES)
    indices_benign = np.random.choice(len(X_benign), n_samples, replace=False)
    indices_attack = np.random.choice(len(X_attack), n_samples, replace=False)
    
    X_benign_downsampled = X_benign[indices_benign]
    y_benign_downsampled = y_benign[indices_benign]
    X_attack_downsampled = X_attack[indices_attack]
    y_attack_downsampled = y_attack[indices_attack]
    
    X_balanced = np.vstack([X_benign_downsampled, X_attack_downsampled])
    y_balanced = np.hstack([y_benign_downsampled, y_attack_downsampled])
    
    print(f"   [+] Balanced dataset: {len(X_balanced):,} samples")
    print(f"     BENIGN: {sum(y_balanced==0):,}")
    print(f"     ATTACK: {sum(y_balanced==1):,}")
    
    # Step 7: Split into train and test
    print("\n7. Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_balanced
    )
    print(f"   [+] Training set: {len(X_train):,} samples")
    print(f"   [+] Test set: {len(X_test):,} samples")
    
    # Step 8: Normalize features
    print("\n8. Normalizing features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"   [+] Features normalized (mean=0, std=1)")
    
    # Save scaler
    joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.pkl'))
    print(f"   [+] Scaler saved to {SAVE_DIR}/scaler.pkl")
    
    input_dim = X_train_scaled.shape[1]
    print(f"\n[OK] Preprocessing complete!")
    print(f"   Final input dimension: {input_dim}")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    print("="*70)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, input_dim, selected_features


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE):
    """Create PyTorch DataLoaders"""
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n[OK] Data loaders created")
    print(f"   Batch size: {batch_size}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, test_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def create_federated_data_splits(X_train_tensor, y_train_tensor, num_clients=CLIENT_NUM_IN_TOTAL):
    """Create federated data splits for multiple clients"""
    print(f"\n[DATA] Preparing federated data splits...")
    
    samples_per_client = len(X_train_tensor) // num_clients
    train_data_local_dict = {}
    train_data_local_num_dict = {}
    
    for client_id in range(num_clients):
        start_idx = client_id * samples_per_client
        end_idx = start_idx + samples_per_client if client_id < num_clients - 1 else len(X_train_tensor)
        
        client_X = X_train_tensor[start_idx:end_idx]
        client_y = y_train_tensor[start_idx:end_idx]
        
        client_dataset = TensorDataset(client_X, client_y)
        client_loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        train_data_local_dict[client_id] = client_loader
        train_data_local_num_dict[client_id] = len(client_dataset)
        
        print(f"   Client {client_id}: {len(client_dataset):,} samples, {len(client_loader)} batches")
    
    print(f"\n   Total federated training samples: {sum(train_data_local_num_dict.values()):,}")
    
    return train_data_local_dict, train_data_local_num_dict

