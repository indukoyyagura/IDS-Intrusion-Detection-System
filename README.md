# Network Intrusion Detection System (IDS)

A comprehensive intrusion detection system using deep learning models with both centralized and federated learning approaches.

## Overview

This project implements three different neural network architectures for network intrusion detection:
- **CNN1D_IDS**: 1D Convolutional Neural Network
- **DenseNN_IDS**: Dense/Fully-Connected Neural Network
- **AutoEncoder_IDS**: AutoEncoder for anomaly detection

Both centralized and federated learning approaches are supported.

## Features

- Multiple deep learning models for intrusion detection
- Centralized training on a single machine
- Federated learning across multiple simulated clients
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1-Score, AUC-ROC)
- Automated visualization generation
- Model checkpointing and saving

## Dataset

The system uses the **CIC-IDS 2017** dataset, which contains various types of network attacks including:
- DDoS attacks
- Port Scanning
- FTP-Patator
- SSH-Patator
- Web attacks

Dataset location: `/home/madhu/yuks/MFEDK_IDS/data/CIC-IDS 2017/`

## Project Structure

```
MFEDK_IDS/
├── config.py                 # Configuration and hyperparameters
├── data_loader.py           # Data loading and preprocessing
├── models.py                # Neural network architectures
├── train_centralized.py     # Centralized training logic
├── train_federated.py       # Federated learning logic
├── evaluate.py              # Model evaluation functions
├── visualize.py             # Visualization generation
├── main.py                  # Main execution script
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── data/                    # Dataset directory
├── saved_models/            # Trained models (auto-created)
└── plots/                   # Generated visualizations (auto-created)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

### Setup

1. Clone or navigate to the project directory:
```bash
cd /home/madhu/yuks/MFEDK_IDS
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

#### Train with Centralized Learning (Default)
```bash
python main.py
```

This will:
- Load and preprocess the CIC-IDS 2017 dataset
- Train all three models (CNN1D, DenseNN, AutoEncoder)
- Evaluate performance on test set
- Save trained models to `saved_models/`
- Generate visualization plots in `plots/`

#### Train with Federated Learning
```bash
python main.py --train-federated
```

#### Train with Both Approaches
```bash
python main.py --train-centralized --train-federated
```

### Command Line Arguments

```bash
python main.py [OPTIONS]
```

**Options:**
- `--train-centralized`: Enable centralized training (default: True)
- `--train-federated`: Enable federated learning (default: False)
- `--use-cuda`: Use CUDA if available (default: True)
- `--no-centralized`: Skip centralized training
- `--no-cuda`: Force CPU usage even if GPU is available

### Examples

1. **Train only centralized models:**
```bash
python main.py --train-centralized
```

2. **Train only federated models:**
```bash
python main.py --no-centralized --train-federated
```

3. **Train both with GPU:**
```bash
python main.py --train-centralized --train-federated --use-cuda
```

4. **Train both on CPU only:**
```bash
python main.py --train-centralized --train-federated --no-cuda
```

## Configuration

Edit `config.py` to customize:

### Training Hyperparameters
```python
# Centralized training
CENTRALIZED_EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# Federated learning
COMM_ROUND = 10              # Communication rounds
LOCAL_EPOCHS = 1             # Local epochs per round
CLIENT_NUM_IN_TOTAL = 5      # Number of clients
```

### Data Settings
```python
BALANCE_SAMPLES = 50000      # Samples per class
TEST_SIZE = 0.2              # Test set proportion
```

### File Paths
```python
DATA_DIR = '/path/to/data/CIC-IDS 2017'
SAVE_DIR = '/path/to/saved_models'
PLOTS_DIR = '/path/to/plots'
```

## Output

### Saved Models

Trained models are saved to `saved_models/` with the following naming convention:
- `cnn1d_ids_model.pth` - Centralized CNN1D model
- `densenn_ids_model.pth` - Centralized DenseNN model
- `autoencoder_ids_model.pth` - Centralized AutoEncoder model
- `cnn1d_ids_federated_model.pth` - Federated CNN1D model
- `densenn_ids_federated_model.pth` - Federated DenseNN model
- `autoencoder_ids_federated_model.pth` - Federated AutoEncoder model
- `scaler.pkl` - StandardScaler for preprocessing

### Visualization Plots

Plots are saved to `plots/` directory:

**Centralized Training:**
- `centralized_training_history.png` - Training curves and metrics
- `centralized_confusion_matrices.png` - Confusion matrices for all models
- `centralized_roc_curves.png` - ROC curves and AUC scores

**Federated Learning:**
- `federated_cnn1d_training.png` - CNN1D federated training results
- `federated_densenn_training.png` - DenseNN federated training results
- `federated_autoencoder_training.png` - AutoEncoder federated training results
- `federated_confusion_matrices.png` - Confusion matrices
- `federated_roc_curves.png` - ROC curves

**Comparison:**
- `centralized_vs_federated_comparison.png` - Side-by-side comparison

### Console Output

The script provides detailed console output including:
- Dataset loading progress
- Preprocessing statistics
- Training progress (epoch-by-epoch)
- Evaluation metrics
- Model saving confirmations
- Final summary

## Model Architectures

### CNN1D_IDS
- 1D Convolutional layers for temporal pattern detection
- Batch normalization and max pooling
- ~589,378 parameters

### DenseNN_IDS
- Fully connected layers with dropout
- Batch normalization between layers
- ~59,842 parameters

### AutoEncoder_IDS
- Encoder-decoder architecture
- Bottleneck dimension: 32
- Hybrid loss (reconstruction + classification)
- ~124,182 parameters

## Federated Learning

The federated learning implementation uses:
- **Algorithm**: FedAvg (Federated Averaging)
- **Clients**: 5 simulated network segments
- **Communication Rounds**: 10
- **Local Epochs**: 1 per round
- **Aggregation**: Weighted averaging by sample count

This approach provides:
- Privacy-preserving training
- Distributed computation
- Scalability to more clients
- Robustness to client failures

## Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **Confusion Matrix**: Detailed classification breakdown

## Expected Results

Typical performance on CIC-IDS 2017:
- CNN1D_IDS: ~99.7% accuracy
- DenseNN_IDS: ~99.5% accuracy
- AutoEncoder_IDS: ~99.0% accuracy

Federated models achieve comparable performance to centralized training while maintaining data privacy.

## Troubleshooting

### Out of Memory Error
If you encounter GPU memory errors:
```bash
python main.py --no-cuda
```

Or reduce batch size in `config.py`:
```python
BATCH_SIZE = 64  # or 32
```

### File Not Found Error
Ensure the dataset is located at:
```
/home/madhu/yuks/MFEDK_IDS/data/CIC-IDS 2017/
```

Or update `DATA_DIR` in `config.py`.

### Import Errors
Reinstall dependencies:
```bash
pip install -r requirements.txt --upgrade
```

### Slow Training
- Enable GPU: `python main.py --use-cuda`
- Reduce epochs in `config.py`
- Use fewer clients for federated learning

## Citation

If you use this code, please cite the CIC-IDS 2017 dataset:

```
Sharafaldin, I., Lashkari, A.H., & Ghorbani, A.A. (2018).
Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.
4th International Conference on Information Systems Security and Privacy (ICISSP).
```

## License

This project is for research and educational purposes.

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.

---

## Quick Start Example

```bash
# Install dependencies
pip install -r requirements.txt

# Run with default settings (centralized training only)
python main.py

# Run with both centralized and federated training
python main.py --train-centralized --train-federated

# Check results
ls saved_models/  # View trained models
ls plots/         # View generated plots
```

## Advanced Usage

### Loading Saved Models

```python
import torch
from models import CNN1D_IDS

# Load model
checkpoint = torch.load('saved_models/cnn1d_ids_model.pth')
model = CNN1D_IDS(input_dim=68)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get training history
history = checkpoint['training_history']
test_accuracy = checkpoint['test_accuracy']
```

### Custom Training Configuration

Create a custom configuration by modifying `config.py` or create a new config file.

## System Requirements

### Minimum Requirements
- CPU: 4 cores
- RAM: 8 GB
- Storage: 5 GB free space
- Python 3.8+

### Recommended Requirements
- CPU: 8+ cores
- RAM: 16 GB
- GPU: NVIDIA GPU with 4+ GB VRAM
- Storage: 10 GB free space
- Python 3.9+
- CUDA 11.0+

## Notes

- First run will take longer due to data preprocessing
- GPU acceleration significantly speeds up training
- Plots are saved automatically; no display required
- All output is logged to console
- Models are saved with complete training history

