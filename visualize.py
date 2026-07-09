"""
Visualization module
Creates and saves comprehensive plots for training analysis
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

from config import PLOTS_DIR, CLIENT_NUM_PER_ROUND, COMM_ROUND


def save_training_history_comparison(histories, model_names, colors, save_path):
    """Save training history comparison plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History Comparison', fontsize=16, fontweight='bold')
    
    # Loss curves
    for model_name, history, color in zip(model_names, histories, colors):
        axes[0, 0].plot(history['train_loss'], label=f"{model_name} Train", color=color, linestyle='-', alpha=0.8)
        axes[0, 0].plot(history['test_loss'], label=f"{model_name} Test", color=color, linestyle='--', alpha=0.8)
    
    axes[0, 0].set_title('Loss over Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)
    
    # Accuracy curves
    for model_name, history, color in zip(model_names, histories, colors):
        axes[0, 1].plot(history['train_acc'], label=f"{model_name} Train", color=color, linestyle='-', alpha=0.8)
        axes[0, 1].plot(history['test_acc'], label=f"{model_name} Test", color=color, linestyle='--', alpha=0.8)
    
    axes[0, 1].set_title('Accuracy over Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)
    
    # Training time comparison
    training_times = [sum(h['epoch_times']) for h in histories]
    axes[0, 2].bar(model_names, training_times, color=colors, alpha=0.7)
    axes[0, 2].set_title('Total Training Time')
    axes[0, 2].set_ylabel('Time (seconds)')
    axes[0, 2].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # Final test accuracy comparison
    final_accs = [h['test_acc'][-1] for h in histories]
    axes[1, 0].bar(model_names, final_accs, color=colors, alpha=0.7)
    axes[1, 0].set_title('Final Test Accuracy')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(final_accs):
        axes[1, 0].text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Learning curves (last 5 epochs)
    for model_name, history, color in zip(model_names, histories, colors):
        if len(history['train_acc']) >= 5:
            last_5_train = history['train_acc'][-5:]
            last_5_test = history['test_acc'][-5:]
            epochs = range(len(history['train_acc'])-5, len(history['train_acc']))
            axes[1, 1].plot(epochs, last_5_train, 'o-', label=f"{model_name} Train", color=color, alpha=0.8)
            axes[1, 1].plot(epochs, last_5_test, 's--', label=f"{model_name} Test", color=color, alpha=0.8)
    
    axes[1, 1].set_title('Convergence (Last 5 Epochs)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)
    
    # Parameter count comparison
    param_counts = [589378, 59842, 124182]  # Approximate values
    axes[1, 2].bar(model_names, [p/1000 for p in param_counts], color=colors, alpha=0.7)
    axes[1, 2].set_title('Model Parameters')
    axes[1, 2].set_ylabel('Parameters (thousands)')
    axes[1, 2].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def save_confusion_matrices(results_dict, save_path):
    """Save confusion matrix plots"""
    
    models = list(results_dict.items())
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
    fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
    
    if len(models) == 1:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(models):
        cm = results['confusion_matrix']
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = axes[idx].imshow(cm_norm, cmap='Blues', interpolation='nearest')
        axes[idx].set_title(f'{model_name}\n(Normalized)', fontsize=12)
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = axes[idx].text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2%})',
                                    ha="center", va="center", 
                                    color="red" if i != j else "black",
                                    fontweight='bold')
        
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['BENIGN', 'ATTACK'])
        axes[idx].set_yticklabels(['BENIGN', 'ATTACK'])
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        
        plt.colorbar(im, ax=axes[idx], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def save_roc_curves(results_dict, save_path):
    """Save ROC curve plots"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('ROC Curves Analysis', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        axes[0].plot(results['fpr'], results['tpr'], 
                    label=f"{model_name} (AUC = {results['roc_auc']:.4f})",
                    color=colors[idx % len(colors)], linewidth=2)
    
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curves - All Models')
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)
    
    # AUC comparison
    model_names = list(results_dict.keys())
    auc_scores = [r['roc_auc'] for r in results_dict.values()]
    
    bars = axes[1].bar(range(len(model_names)), auc_scores, 
                      color=colors[:len(model_names)], alpha=0.7)
    axes[1].set_title('AUC-ROC Comparison')
    axes[1].set_ylabel('AUC Score')
    axes[1].set_ylim([0.5, 1.0])
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=0.5, color='r', linestyle='--', label='Random', alpha=0.5)
    
    for i, (bar, score) in enumerate(zip(bars, auc_scores)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def save_federated_training_results(history, model_name, save_path):
    """Save federated training results plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Federated Training Results - {model_name}', fontsize=16, fontweight='bold')
    
    # Global loss
    axes[0,0].plot(range(1, len(history['global_losses']) + 1), 
                   history['global_losses'], 'b-o', linewidth=2, markersize=6)
    axes[0,0].set_title('Global Loss Over Communication Rounds')
    axes[0,0].set_xlabel('Communication Round')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].grid(True, alpha=0.3)
    
    # Client-specific losses
    colors = plt.cm.tab10(np.linspace(0, 1, CLIENT_NUM_PER_ROUND))
    for client_idx in range(CLIENT_NUM_PER_ROUND):
        client_losses = history['client_losses'][client_idx]
        if client_losses:
            axes[0,1].plot(range(1, len(client_losses) + 1), client_losses, 
                           'o-', label=f'Client {client_idx}', 
                           color=colors[client_idx], alpha=0.8)
    
    axes[0,1].set_title('Client-Specific Losses')
    axes[0,1].set_xlabel('Communication Round')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Test accuracy
    axes[1,0].plot(range(1, len(history['test_accuracy']) + 1), 
                   history['test_accuracy'], 'g-o', linewidth=2, markersize=6)
    axes[1,0].set_title('Test Accuracy Over Communication Rounds')
    axes[1,0].set_xlabel('Communication Round')
    axes[1,0].set_ylabel('Accuracy (%)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Training time
    axes[1,1].bar(range(1, len(history['round_times']) + 1), 
                  history['round_times'], alpha=0.7, color='green')
    axes[1,1].set_title('Training Time per Round')
    axes[1,1].set_xlabel('Communication Round')
    axes[1,1].set_ylabel('Time (seconds)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def save_centralized_vs_federated_comparison(cent_results, fed_results, 
                                            cent_histories, fed_histories, save_path):
    """Save centralized vs federated comparison plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Centralized vs Federated Learning Comparison', fontsize=16, fontweight='bold')
    
    # Accuracy comparison
    model_names = list(cent_results.keys())
    cent_accs = [r['accuracy'] * 100 for r in cent_results.values()]
    fed_accs = [r['accuracy'] * 100 for r in fed_results.values()]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, cent_accs, width, label='Centralized', alpha=0.7, color='steelblue')
    axes[0, 0].bar(x + width/2, fed_accs, width, label='Federated', alpha=0.7, color='orange')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Final Accuracy Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    for i, (c, f) in enumerate(zip(cent_accs, fed_accs)):
        axes[0, 0].text(i - width/2, c + 0.5, f'{c:.1f}%', ha='center', va='bottom', fontsize=8)
        axes[0, 0].text(i + width/2, f + 0.5, f'{f:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # F1-Score comparison
    cent_f1 = [r['f1'] * 100 for r in cent_results.values()]
    fed_f1 = [r['f1'] * 100 for r in fed_results.values()]
    
    axes[0, 1].bar(x - width/2, cent_f1, width, label='Centralized', alpha=0.7, color='steelblue')
    axes[0, 1].bar(x + width/2, fed_f1, width, label='Federated', alpha=0.7, color='orange')
    axes[0, 1].set_ylabel('F1-Score (%)')
    axes[0, 1].set_title('F1-Score Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Training time comparison
    cent_times = [sum(h['epoch_times']) for h in cent_histories.values()]
    fed_times = [sum(h['round_times']) for h in fed_histories.values()]
    
    axes[1, 0].bar(x - width/2, cent_times, width, label='Centralized', alpha=0.7, color='steelblue')
    axes[1, 0].bar(x + width/2, fed_times, width, label='Federated', alpha=0.7, color='orange')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].set_title('Training Time Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Convergence comparison for federated
    colors = ['blue', 'green', 'red']
    for idx, (model_name, hist_fed) in enumerate(fed_histories.items()):
        axes[1, 1].plot(range(1, len(hist_fed['test_accuracy']) + 1), 
                       hist_fed['test_accuracy'], 
                       'o-', label=model_name, color=colors[idx], linewidth=2, alpha=0.7)
    
    axes[1, 1].set_xlabel('Communication Round')
    axes[1, 1].set_ylabel('Test Accuracy (%)')
    axes[1, 1].set_title('Federated Learning Convergence')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")

