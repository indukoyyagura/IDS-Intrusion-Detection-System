# IDS Detection System - Usage Guide

## Quick Start

### Option 1: Using the Shell Script (Recommended)

```bash
# Make script executable (first time only)
chmod +x run_ids.sh

# Run centralized training only
./run_ids.sh centralized

# Run federated learning only
./run_ids.sh federated

# Run both centralized and federated
./run_ids.sh both
```

### Option 2: Using Python Directly

```bash
# Centralized training only (default)
python main.py

# Federated learning only
python main.py --no-centralized --train-federated

# Both centralized and federated
python main.py --train-centralized --train-federated
```

## Detailed Command Options

### Main Script Arguments

```bash
python main.py [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--train-centralized` | Enable centralized training | True |
| `--train-federated` | Enable federated learning | False |
| `--use-cuda` | Use GPU if available | True |
| `--no-centralized` | Disable centralized training | - |
| `--no-cuda` | Force CPU usage | - |

## Examples

### Example 1: Train All Models (Centralized)

This is the default behavior and trains all three models using centralized learning.

```bash
python main.py
```

**What happens:**
1. Loads CIC-IDS 2017 dataset
2. Preprocesses data (balancing, normalization)
3. Trains CNN1D_IDS model (15 epochs)
4. Trains DenseNN_IDS model (15 epochs)
5. Trains AutoEncoder_IDS model (15 epochs)
6. Evaluates all models on test set
7. Saves models to `saved_models/`
8. Generates plots in `plots/`

**Expected output files:**
- `saved_models/cnn1d_ids_model.pth`
- `saved_models/densenn_ids_model.pth`
- `saved_models/autoencoder_ids_model.pth`
- `saved_models/scaler.pkl`
- `plots/centralized_training_history.png`
- `plots/centralized_confusion_matrices.png`
- `plots/centralized_roc_curves.png`

**Estimated time:** ~5-10 minutes on GPU, ~30-60 minutes on CPU

---

### Example 2: Train with Federated Learning

Train models using federated learning across 5 simulated clients.

```bash
python main.py --no-centralized --train-federated
```

**What happens:**
1. Loads and preprocesses dataset
2. Splits data among 5 clients
3. Trains each model using FedAvg (10 communication rounds)
4. Evaluates federated models
5. Saves federated models
6. Generates federated training plots

**Expected output files:**
- `saved_models/cnn1d_ids_federated_model.pth`
- `saved_models/densenn_ids_federated_model.pth`
- `saved_models/autoencoder_ids_federated_model.pth`
- `plots/federated_cnn1d_training.png`
- `plots/federated_densenn_training.png`
- `plots/federated_autoencoder_training.png`
- `plots/federated_confusion_matrices.png`
- `plots/federated_roc_curves.png`

**Estimated time:** ~3-7 minutes on GPU, ~20-40 minutes on CPU

---

### Example 3: Train Both Approaches for Comparison

Train using both centralized and federated learning to compare results.

```bash
python main.py --train-centralized --train-federated
```

or

```bash
./run_ids.sh both
```

**Additional output:**
- `plots/centralized_vs_federated_comparison.png`

This generates a comprehensive comparison plot showing:
- Accuracy comparison
- F1-Score comparison
- Training time comparison
- Convergence curves

**Estimated time:** ~8-17 minutes on GPU, ~50-100 minutes on CPU

---

### Example 4: CPU-Only Training

Force CPU usage even if GPU is available (useful for debugging or limited GPU memory).

```bash
python main.py --no-cuda
```

---

## Understanding the Output

### Console Output

The script provides detailed progress information:

```
======================================================================
Network Intrusion Detection System - IDS Detection
======================================================================

[DEVICE] Using: cuda

======================================================================
STEP 1: DATA LOADING AND PREPROCESSING
======================================================================
[LOAD] Loading datasets...
[+] Monday: 529,918 samples
[+] Tuesday: 445,909 samples
...

[CONFIG] Data Preprocessing Pipeline
======================================================================
1. Creating binary labels (BENIGN vs ATTACK)...
2. Handling infinite values...
...

======================================================================
STEP 2: CENTRALIZED TRAINING
======================================================================

[MODELS] Created models:
   CNN1D_IDS:        589,378 parameters
   DenseNN_IDS:       59,842 parameters
   AutoEncoder_IDS:  124,182 parameters

======================================================================
>> Training CNN1D_IDS
======================================================================
Epoch [ 1/15] | Train Loss: 0.0401 | Train Acc: 98.50% | ...
...

[INFO] Evaluating CNN1D_IDS...
  [+] Accuracy:  99.77%
  [+] Precision: 99.64%
  [+] Recall:    99.89%
  [+] F1-Score:  99.77%
  [+] AUC-ROC:   0.9999
```

### Performance Metrics Explained

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Of predicted attacks, how many were real |
| **Recall** | TP / (TP + FN) | Of actual attacks, how many were detected |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Balanced metric |
| **AUC-ROC** | Area under ROC curve | Overall discriminative ability |

Where:
- TP = True Positives (attacks correctly identified)
- TN = True Negatives (benign correctly identified)
- FP = False Positives (benign incorrectly flagged)
- FN = False Negatives (attacks missed)

---

## Customizing Configuration

Edit `config.py` to customize training parameters:

### Change Training Duration

```python
# Centralized training
CENTRALIZED_EPOCHS = 20  # Increase from 15 to 20

# Federated learning
COMM_ROUND = 15  # Increase from 10 to 15
```

### Adjust Batch Size (for memory constraints)

```python
BATCH_SIZE = 64  # Decrease from 128
```

### Modify Federated Learning Setup

```python
CLIENT_NUM_IN_TOTAL = 10  # Increase clients from 5 to 10
LOCAL_EPOCHS = 2  # Increase local training
```

### Change Data Paths

```python
DATA_DIR = '/path/to/your/data/CIC-IDS 2017'
SAVE_DIR = '/path/to/save/models'
PLOTS_DIR = '/path/to/save/plots'
```

---

## Viewing Results

### Saved Models

Models are saved in PyTorch format (`.pth` files):

```bash
ls -lh saved_models/
```

To load a model:

```python
import torch
from models import CNN1D_IDS

checkpoint = torch.load('saved_models/cnn1d_ids_model.pth')
model = CNN1D_IDS(input_dim=68)
model.load_state_dict(checkpoint['model_state_dict'])

# Access training info
print(f"Test Accuracy: {checkpoint['test_accuracy']}")
print(f"Training History: {checkpoint['training_history']}")
```

### Generated Plots

All plots are saved as high-resolution PNG files (300 DPI):

```bash
ls -lh plots/
```

View plots using any image viewer:
```bash
# Linux
xdg-open plots/centralized_training_history.png

# Or use your preferred image viewer
eog plots/*.png
```

---

## Troubleshooting

### Problem: Out of Memory Error

**Solution 1:** Use CPU instead
```bash
python main.py --no-cuda
```

**Solution 2:** Reduce batch size in `config.py`
```python
BATCH_SIZE = 64  # or even 32
```

**Solution 3:** Train models one at a time
Modify `main.py` to comment out models you don't need.

---

### Problem: Dataset Not Found

**Error:** `FileNotFoundError: [Errno 2] No such file or directory`

**Solution:** Verify dataset location
```bash
ls -la /home/madhu/yuks/MFEDK_IDS/data/CIC-IDS\ 2017/
```

If files are in a different location, update `DATA_DIR` in `config.py`.

---

### Problem: Slow Training

**Solution 1:** Enable GPU
```bash
python main.py --use-cuda
```

**Solution 2:** Reduce epochs
Edit `config.py`:
```python
CENTRALIZED_EPOCHS = 5  # Instead of 15
COMM_ROUND = 5  # Instead of 10
```

**Solution 3:** Use fewer clients for federated learning
```python
CLIENT_NUM_IN_TOTAL = 3
```

---

### Problem: Import Errors

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

If that doesn't work, install manually:
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn joblib
```

---

## Advanced Usage

### Training Only Specific Models

To train only specific models, modify `main.py` and comment out unwanted training calls.

### Custom Data Loading

To use a different dataset:

1. Modify `load_and_combine_datasets()` in `data_loader.py`
2. Ensure data format matches (CSV with numerical features and a label column)
3. Update preprocessing as needed

### Hyperparameter Tuning

Create a script for grid search:

```python
from config import *
import itertools

# Define hyperparameter grid
learning_rates = [0.001, 0.0001]
batch_sizes = [64, 128, 256]

for lr, bs in itertools.product(learning_rates, batch_sizes):
    print(f"Training with lr={lr}, batch_size={bs}")
    # Run training with modified parameters
    # Save results
```

---

## Performance Benchmarks

Expected performance on CIC-IDS 2017 dataset:

| Model | Approach | Accuracy | F1-Score | Training Time (GPU) |
|-------|----------|----------|----------|---------------------|
| CNN1D_IDS | Centralized | 99.7% | 99.7% | ~3 min |
| CNN1D_IDS | Federated | 99.6% | 99.6% | ~2 min |
| DenseNN_IDS | Centralized | 99.5% | 99.5% | ~2 min |
| DenseNN_IDS | Federated | 99.4% | 99.4% | ~1 min |
| AutoEncoder_IDS | Centralized | 99.0% | 99.0% | ~2.5 min |
| AutoEncoder_IDS | Federated | 98.9% | 98.9% | ~2 min |

*Times measured on NVIDIA GPU. CPU times are approximately 5-10x longer.*

---

## Next Steps

After training:

1. **Analyze results:** Review generated plots in `plots/` directory
2. **Compare approaches:** Look at centralized vs federated comparison
3. **Test on new data:** Use saved models for inference
4. **Deploy model:** Export best model for production use
5. **Fine-tune:** Adjust hyperparameters in `config.py` for better results

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review console output for error messages
3. Verify dataset location and format
4. Check system requirements in README.md

