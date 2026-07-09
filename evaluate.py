"""
Model evaluation module
Provides comprehensive evaluation metrics for all models
"""

import torch
from config import LOCAL_EPOCHS, CLIENT_NUM_IN_TOTAL
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_curve, auc)


def evaluate_model(model, test_loader, model_name, device, is_autoencoder=False):
    """Comprehensive model evaluation"""
    
    print(f"\n[INFO] Evaluating {model_name}...")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            if is_autoencoder:
                _, _, outputs = model(batch_x)
            else:
                outputs = model(batch_x)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    cm = confusion_matrix(all_labels, all_preds)
    
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    print(f"  [+] Accuracy:  {accuracy*100:.2f}%")
    print(f"  [+] Precision: {precision*100:.2f}%")
    print(f"  [+] Recall:    {recall*100:.2f}%")
    print(f"  [+] F1-Score:  {f1*100:.2f}%")
    print(f"  [+] AUC-ROC:   {roc_auc:.4f}")
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }


def print_comparison_table(results_dict):
    """Print comparison table for all models"""
    
    print("\n" + "="*90)
    print("[DATA] Model Performance Comparison")
    print("="*90)
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'AUC-ROC':<10}")
    print("-"*90)
    
    for model_name, results in results_dict.items():
        print(f"{model_name:<20} | "
              f"{results['accuracy']*100:>9.2f}% | "
              f"{results['precision']*100:>9.2f}% | "
              f"{results['recall']*100:>9.2f}% | "
              f"{results['f1']*100:>9.2f}% | "
              f"{results['roc_auc']:>9.4f}")
    
    print("="*90)
    
    # Find best model
    best_model = max(results_dict.items(), key=lambda x: x[1]['f1'])[0]
    print(f"\n[BEST] Best Model (by F1-Score): {best_model}")


def print_confusion_matrix_analysis(results_dict):
    """Print detailed confusion matrix analysis"""
    
    print("\n" + "="*70)
    print("[DATA] Confusion Matrix Analysis")
    print("="*70)
    
    for model_name, results in results_dict.items():
        cm = results['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n{model_name}:")
        print(f"  True Negatives:  {tn:>6,} (Correctly identified benign)")
        print(f"  False Positives: {fp:>6,} (Benign misclassified as attack)")
        print(f"  False Negatives: {fn:>6,} (Attack missed)")
        print(f"  True Positives:  {tp:>6,} (Correctly detected attacks)")
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"  Specificity: {specificity*100:.2f}% (True Negative Rate)")
        print(f"  Sensitivity: {sensitivity*100:.2f}% (True Positive Rate)")
    
    print("="*70)


def save_model_checkpoint(model, model_name, input_dim, results, history, save_dir, 
                         is_federated=False, is_autoencoder=False):
    """Save model checkpoint with metadata"""
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'input_dim': input_dim,
        'test_accuracy': results['accuracy'],
        'test_f1': results['f1'],
        'training_history': history
    }
    
    if is_autoencoder:
        checkpoint['bottleneck_dim'] = 32
    
    if is_federated:
        checkpoint['federated_config'] = {
            'comm_rounds': history.get('global_losses', []),
            'local_epochs': LOCAL_EPOCHS,
            'num_clients': CLIENT_NUM_IN_TOTAL,
            'aggregation': 'FedAvg'
        }
        filename = f"{model_name.lower().replace(' ', '_')}_federated_model.pth"
    else:
        filename = f"{model_name.lower().replace(' ', '_')}_model.pth"
    
    import os
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"   [+] Saved {model_name} to {filepath}")
    
    return filepath

