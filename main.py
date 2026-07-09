"""
Main entry point for IDS Detection System
Orchestrates data loading, training, evaluation, and visualization
"""

import os
import argparse
import torch
import warnings
warnings.filterwarnings('ignore')

from config import *
from data_loader import (load_and_combine_datasets, preprocess_data, 
                        create_dataloaders, create_federated_data_splits)
from models import CNN1D_IDS, DenseNN_IDS, AutoEncoder_IDS
from train_centralized import train_classification_model, train_autoencoder
from train_federated import train_federated_model
from evaluate import (evaluate_model, print_comparison_table, 
                     print_confusion_matrix_analysis, save_model_checkpoint)
from visualize import (save_training_history_comparison, save_confusion_matrices,
                      save_roc_curves, save_federated_training_results,
                      save_centralized_vs_federated_comparison)


def main(args):
    """Main execution function"""
    
    print("="*90)
    print("Network Intrusion Detection System - IDS Detection")
    print("="*90)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"\n[DEVICE] Using: {device}")
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # ============================================================
    # STEP 1: DATA LOADING AND PREPROCESSING
    # ============================================================
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*70)
    
    df = load_and_combine_datasets()
    X_train, X_test, y_train, y_test, input_dim, selected_features = preprocess_data(df)
    train_loader, test_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = create_dataloaders(
        X_train, y_train, X_test, y_test
    )
    
    # ============================================================
    # STEP 2: CENTRALIZED TRAINING
    # ============================================================
    if args.train_centralized:
        print("\n" + "="*70)
        print("STEP 2: CENTRALIZED TRAINING")
        print("="*70)
        
        # Initialize models
        model_cnn = CNN1D_IDS(input_dim).to(device)
        model_dense = DenseNN_IDS(input_dim).to(device)
        model_ae = AutoEncoder_IDS(input_dim, bottleneck_dim=AUTOENCODER_BOTTLENECK_DIM).to(device)
        
        print(f"\n[MODELS] Created models:")
        print(f"   CNN1D_IDS:       {sum(p.numel() for p in model_cnn.parameters()):>8,} parameters")
        print(f"   DenseNN_IDS:     {sum(p.numel() for p in model_dense.parameters()):>8,} parameters")
        print(f"   AutoEncoder_IDS: {sum(p.numel() for p in model_ae.parameters()):>8,} parameters")
        
        # Train models
        history_cnn = train_classification_model(
            model_cnn, train_loader, test_loader, "CNN1D_IDS", device
        )
        
        history_dense = train_classification_model(
            model_dense, train_loader, test_loader, "DenseNN_IDS", device
        )
        
        history_ae = train_autoencoder(
            model_ae, train_loader, test_loader, device
        )
        
        # Evaluate models
        print("\n" + "="*70)
        print("CENTRALIZED MODEL EVALUATION")
        print("="*70)
        
        results_cnn = evaluate_model(model_cnn, test_loader, "CNN1D_IDS", device, is_autoencoder=False)
        results_dense = evaluate_model(model_dense, test_loader, "DenseNN_IDS", device, is_autoencoder=False)
        results_ae = evaluate_model(model_ae, test_loader, "AutoEncoder_IDS", device, is_autoencoder=True)
        
        centralized_results = {
            'CNN1D_IDS': results_cnn,
            'DenseNN_IDS': results_dense,
            'AutoEncoder_IDS': results_ae
        }
        
        centralized_histories = {
            'CNN1D_IDS': history_cnn,
            'DenseNN_IDS': history_dense,
            'AutoEncoder_IDS': history_ae
        }
        
        print_comparison_table(centralized_results)
        print_confusion_matrix_analysis(centralized_results)
        
        # Save models
        print("\n[SAVE] Saving centralized models...")
        save_model_checkpoint(model_cnn, "CNN1D_IDS", input_dim, results_cnn, history_cnn, SAVE_DIR)
        save_model_checkpoint(model_dense, "DenseNN_IDS", input_dim, results_dense, history_dense, SAVE_DIR)
        save_model_checkpoint(model_ae, "AutoEncoder_IDS", input_dim, results_ae, history_ae, SAVE_DIR, is_autoencoder=True)
        
        # Visualizations
        print("\n[PLOTS] Generating centralized training visualizations...")
        save_training_history_comparison(
            [history_cnn, history_dense, history_ae],
            ['CNN1D_IDS', 'DenseNN_IDS', 'AutoEncoder_IDS'],
            ['blue', 'green', 'red'],
            os.path.join(PLOTS_DIR, 'centralized_training_history.png')
        )
        save_confusion_matrices(
            centralized_results,
            os.path.join(PLOTS_DIR, 'centralized_confusion_matrices.png')
        )
        save_roc_curves(
            centralized_results,
            os.path.join(PLOTS_DIR, 'centralized_roc_curves.png')
        )
    
    # ============================================================
    # STEP 3: FEDERATED LEARNING
    # ============================================================
    if args.train_federated:
        print("\n" + "="*70)
        print("STEP 3: FEDERATED LEARNING")
        print("="*70)
        
        # Prepare federated data splits
        train_data_local_dict, train_data_local_num_dict = create_federated_data_splits(
            X_train_tensor, y_train_tensor
        )
        
        # Train federated models
        model_cnn_fed, history_cnn_fed = train_federated_model(
            CNN1D_IDS, "CNN1D_IDS", input_dim, train_data_local_dict, 
            train_data_local_num_dict, test_loader, device, is_autoencoder=False
        )
        
        model_dense_fed, history_dense_fed = train_federated_model(
            DenseNN_IDS, "DenseNN_IDS", input_dim, train_data_local_dict, 
            train_data_local_num_dict, test_loader, device, is_autoencoder=False
        )
        
        model_ae_fed, history_ae_fed = train_federated_model(
            AutoEncoder_IDS, "AutoEncoder_IDS", input_dim, train_data_local_dict, 
            train_data_local_num_dict, test_loader, device, is_autoencoder=True
        )
        
        # Evaluate federated models
        print("\n" + "="*70)
        print("FEDERATED MODEL EVALUATION")
        print("="*70)
        
        results_cnn_fed = evaluate_model(model_cnn_fed, test_loader, "CNN1D_IDS (Federated)", device, is_autoencoder=False)
        results_dense_fed = evaluate_model(model_dense_fed, test_loader, "DenseNN_IDS (Federated)", device, is_autoencoder=False)
        results_ae_fed = evaluate_model(model_ae_fed, test_loader, "AutoEncoder_IDS (Federated)", device, is_autoencoder=True)
        
        federated_results = {
            'CNN1D_IDS': results_cnn_fed,
            'DenseNN_IDS': results_dense_fed,
            'AutoEncoder_IDS': results_ae_fed
        }
        
        federated_histories = {
            'CNN1D_IDS': history_cnn_fed,
            'DenseNN_IDS': history_dense_fed,
            'AutoEncoder_IDS': history_ae_fed
        }
        
        print_comparison_table(federated_results)
        print_confusion_matrix_analysis(federated_results)
        
        # Save federated models
        print("\n[SAVE] Saving federated models...")
        save_model_checkpoint(model_cnn_fed, "CNN1D_IDS", input_dim, results_cnn_fed, history_cnn_fed, SAVE_DIR, is_federated=True)
        save_model_checkpoint(model_dense_fed, "DenseNN_IDS", input_dim, results_dense_fed, history_dense_fed, SAVE_DIR, is_federated=True)
        save_model_checkpoint(model_ae_fed, "AutoEncoder_IDS", input_dim, results_ae_fed, history_ae_fed, SAVE_DIR, is_federated=True, is_autoencoder=True)
        
        # Visualizations
        print("\n[PLOTS] Generating federated learning visualizations...")
        save_federated_training_results(
            history_cnn_fed, "CNN1D_IDS",
            os.path.join(PLOTS_DIR, 'federated_cnn1d_training.png')
        )
        save_federated_training_results(
            history_dense_fed, "DenseNN_IDS",
            os.path.join(PLOTS_DIR, 'federated_densenn_training.png')
        )
        save_federated_training_results(
            history_ae_fed, "AutoEncoder_IDS",
            os.path.join(PLOTS_DIR, 'federated_autoencoder_training.png')
        )
        save_confusion_matrices(
            federated_results,
            os.path.join(PLOTS_DIR, 'federated_confusion_matrices.png')
        )
        save_roc_curves(
            federated_results,
            os.path.join(PLOTS_DIR, 'federated_roc_curves.png')
        )
        
        # Comparison plots (if both centralized and federated were run)
        if args.train_centralized:
            print("\n[PLOTS] Generating centralized vs federated comparison...")
            save_centralized_vs_federated_comparison(
                centralized_results, federated_results,
                centralized_histories, federated_histories,
                os.path.join(PLOTS_DIR, 'centralized_vs_federated_comparison.png')
            )
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "="*90)
    print("TRAINING COMPLETE")
    print("="*90)
    print(f"\n[SAVED MODELS] Location: {SAVE_DIR}")
    if os.path.exists(SAVE_DIR):
        models = [f for f in os.listdir(SAVE_DIR) if f.endswith('.pth')]
        for model_file in models:
            print(f"   - {model_file}")
    
    print(f"\n[PLOTS] Location: {PLOTS_DIR}")
    if os.path.exists(PLOTS_DIR):
        plots = [f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')]
        for plot_file in plots:
            print(f"   - {plot_file}")
    
    print("\n" + "="*90)
    print("All tasks completed successfully!")
    print("="*90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IDS Detection System - Centralized and Federated Learning')
    
    parser.add_argument('--train-centralized', action='store_true', default=True,
                       help='Train models using centralized learning (default: True)')
    parser.add_argument('--train-federated', action='store_true', default=False,
                       help='Train models using federated learning (default: False)')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                       help='Use CUDA if available (default: True)')
    parser.add_argument('--no-centralized', dest='train_centralized', action='store_false',
                       help='Skip centralized training')
    parser.add_argument('--no-cuda', dest='use_cuda', action='store_false',
                       help='Do not use CUDA even if available')
    
    args = parser.parse_args()
    
    # If both are False (explicitly disabled), enable centralized by default
    if not args.train_centralized and not args.train_federated:
        args.train_centralized = True
    
    main(args)
