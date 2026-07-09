# IDS Detection System - Project Structure

## Overview

This document provides a comprehensive overview of all files in the IDS Detection System and their purposes.

## Main Python Files

### 1. `config.py`
**Purpose:** Central configuration file containing all hyperparameters and settings

**Key Contents:**
- Data paths (DATA_DIR, SAVE_DIR, PLOTS_DIR)
- Data preprocessing parameters
- Training hyperparameters (epochs, batch size, learning rate)
- Federated learning configuration
- Model architecture parameters
- Random seed initialization

**When to modify:**
- Change training duration
- Adjust batch size for memory
- Modify file paths
- Tune hyperparameters

---

### 2. `data_loader.py`
**Purpose:** Data loading, preprocessing, and preparation

**Key Functions:**
- `load_and_combine_datasets()`: Loads CIC-IDS 2017 dataset files
- `preprocess_data()`: Complete preprocessing pipeline
- `create_dataloaders()`: Creates PyTorch DataLoaders
- `create_federated_data_splits()`: Splits data for federated learning

**Processing Steps:**
1. Load multiple CSV files
2. Create binary labels (BENIGN vs ATTACK)
3. Handle infinite/missing values
4. Remove low-variance features
5. Balance dataset
6. Train/test split
7. Feature normalization (StandardScaler)

---

### 3. `models.py`
**Purpose:** Neural network model architectures

**Model Classes:**

#### CNN1D_IDS
- 1D Convolutional Neural Network
- 3 conv layers with batch normalization
- Max pooling and dropout
- Parameters: ~589,378

#### DenseNN_IDS
- Fully connected neural network
- 4 dense layers with batch normalization
- Dropout for regularization
- Parameters: ~59,842

#### AutoEncoder_IDS
- Encoder-decoder architecture
- Bottleneck dimension: 32
- Separate classifier head
- Parameters: ~124,182

**Helper Functions:**
- `create_models()`: Initializes all three models

---

### 4. `train_centralized.py`
**Purpose:** Centralized (traditional) training logic

**Key Functions:**

#### `train_classification_model()`
Trains CNN1D or DenseNN models
- Uses CrossEntropyLoss
- Adam optimizer
- ReduceLROnPlateau scheduler
- Tracks training/validation metrics

#### `train_autoencoder()`
Trains AutoEncoder with hybrid loss
- Combined reconstruction + classification loss
- Weighted by alpha parameter (default 0.7)
- Tracks multiple loss components

**Features:**
- Automatic best model selection
- Epoch-by-epoch progress logging
- Training history tracking

---

### 5. `train_federated.py`
**Purpose:** Federated learning implementation

**Key Functions:**

#### `train_local_model_classification()`
Local training for CNN1D/DenseNN on single client
- Trains for LOCAL_EPOCHS
- Returns model state and loss

#### `train_local_model_autoencoder()`
Local training for AutoEncoder
- Hybrid loss function
- Returns model state and loss

#### `federated_averaging()`
FedAvg aggregation algorithm
- Weighted averaging by sample count
- Combines client models into global model

#### `train_federated_model()`
Main federated training loop
- Manages communication rounds
- Coordinates client training
- Performs aggregation
- Evaluates global model

**Federated Process:**
1. Initialize global model
2. For each communication round:
   - Distribute model to clients
   - Clients train locally
   - Aggregate updates (FedAvg)
   - Update global model
   - Evaluate on test set

---

### 6. `evaluate.py`
**Purpose:** Model evaluation and performance metrics

**Key Functions:**

#### `evaluate_model()`
Comprehensive evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix
- ROC curve and AUC
- Returns detailed results dictionary

#### `print_comparison_table()`
Formatted comparison of multiple models
- Side-by-side metrics
- Identifies best model

#### `print_confusion_matrix_analysis()`
Detailed confusion matrix breakdown
- True/False Positives/Negatives
- Specificity and Sensitivity

#### `save_model_checkpoint()`
Saves trained model with metadata
- Model state dictionary
- Training history
- Performance metrics
- Configuration info

---

### 7. `visualize.py`
**Purpose:** Generate and save visualization plots

**Key Functions:**

#### `save_training_history_comparison()`
Creates comprehensive training plots:
- Loss curves (train/test)
- Accuracy curves
- Training time comparison
- Final accuracy bars
- Convergence analysis
- Parameter counts

#### `save_confusion_matrices()`
Generates confusion matrix heatmaps:
- Normalized matrices
- Absolute counts
- Color-coded visualization

#### `save_roc_curves()`
Creates ROC analysis plots:
- ROC curves for all models
- AUC comparison bar chart
- Random classifier baseline

#### `save_federated_training_results()`
Federated learning visualizations:
- Global loss over rounds
- Client-specific losses
- Test accuracy progression
- Training time per round

#### `save_centralized_vs_federated_comparison()`
Comparative analysis:
- Accuracy comparison
- F1-Score comparison
- Training time comparison
- Convergence curves

**Features:**
- Uses non-interactive backend (Agg)
- High-resolution (300 DPI)
- Automatically saves to PLOTS_DIR
- No display required

---

### 8. `main.py`
**Purpose:** Main entry point and orchestration

**Workflow:**
1. Parse command-line arguments
2. Initialize device (CPU/GPU)
3. Load and preprocess data
4. (Optional) Centralized training
   - Train all three models
   - Evaluate performance
   - Save models
   - Generate plots
5. (Optional) Federated learning
   - Create data splits
   - Train federated models
   - Evaluate performance
   - Save models
   - Generate plots
6. (If both) Generate comparison plots
7. Print final summary

**Command-line Interface:**
- `--train-centralized`: Enable centralized
- `--train-federated`: Enable federated
- `--use-cuda`: Use GPU
- `--no-centralized`: Disable centralized
- `--no-cuda`: Force CPU

---

## Helper Files

### `run_ids.sh`
Bash script for easy execution
- Checks Python installation
- Verifies dependencies
- Provides simple interface
- Three modes: centralized, federated, both

**Usage:**
```bash
./run_ids.sh [centralized|federated|both]
```

---

### `requirements.txt`
Python package dependencies
- Core: numpy, pandas, matplotlib, seaborn
- ML: scikit-learn, joblib
- Deep Learning: torch, torchvision
- Utilities: tqdm

---

## Documentation Files

### `README.md`
**Main documentation covering:**
- Project overview
- Features
- Installation instructions
- Basic usage examples
- Configuration guide
- Output description
- Model architectures
- Troubleshooting
- Performance metrics
- System requirements

---

### `USAGE.md`
**Detailed usage guide with:**
- Quick start instructions
- Command-line options
- Multiple examples
- Output interpretation
- Performance metrics explanation
- Customization guide
- Troubleshooting solutions
- Advanced usage
- Performance benchmarks

---

### `PROJECT_STRUCTURE.md`
**This file** - Complete project structure documentation

---

## Directory Structure

```
MFEDK_IDS/
│
├── Core Python Modules
│   ├── config.py              # Configuration
│   ├── data_loader.py         # Data handling
│   ├── models.py              # Model architectures
│   ├── train_centralized.py  # Centralized training
│   ├── train_federated.py    # Federated training
│   ├── evaluate.py            # Evaluation metrics
│   ├── visualize.py           # Plot generation
│   └── main.py                # Main entry point
│
├── Helper Scripts
│   ├── run_ids.sh             # Shell script launcher
│   └── requirements.txt       # Dependencies
│
├── Documentation
│   ├── README.md              # Main documentation
│   ├── USAGE.md               # Usage guide
│   └── PROJECT_STRUCTURE.md  # This file
│
├── Data Directory (auto-created)
│   └── data/
│       └── CIC-IDS 2017/      # Dataset location
│
├── Output Directories (auto-created)
│   ├── saved_models/          # Trained models
│   │   ├── *.pth             # PyTorch models
│   │   └── scaler.pkl        # Preprocessor
│   └── plots/                 # Visualizations
│       └── *.png             # Generated plots
│
└── Original Notebook
    └── IDS_Detection_Complete.ipynb  # Source notebook
```

---

## File Dependencies

### Dependency Graph

```
main.py
├── config.py (settings)
├── data_loader.py
│   └── config.py
├── models.py
│   └── config.py
├── train_centralized.py
│   ├── config.py
│   └── models.py
├── train_federated.py
│   ├── config.py
│   └── models.py
├── evaluate.py
│   ├── config.py
│   └── models.py
└── visualize.py
    └── config.py
```

All modules depend on `config.py` for settings.

---

## Data Flow

### Centralized Training Flow

```
Data Files (CSV)
    ↓
load_and_combine_datasets()
    ↓
preprocess_data()
    ↓
create_dataloaders()
    ↓
train_classification_model() / train_autoencoder()
    ↓
evaluate_model()
    ↓
save_model_checkpoint()
    ↓
save_*_plots()
```

### Federated Learning Flow

```
Preprocessed Data
    ↓
create_federated_data_splits()
    ↓
train_federated_model()
    ├── train_local_model_*() (per client)
    ├── federated_averaging()
    └── evaluate (per round)
    ↓
evaluate_model()
    ↓
save_model_checkpoint()
    ↓
save_federated_*_plots()
```

---

## Key Design Decisions

### 1. Modular Architecture
Each component has a single responsibility:
- Easy to test
- Simple to modify
- Clear organization

### 2. Configuration Centralization
All settings in `config.py`:
- Single source of truth
- Easy customization
- Consistent across modules

### 3. Automatic Directory Creation
Directories created automatically:
- No manual setup needed
- Prevents errors

### 4. No Emoji in Code
Clean, professional code:
- Better for logs
- Suitable for production
- Easier to parse

### 5. Plot Saving (Not Display)
Plots saved to files:
- Works in headless environments
- High resolution for papers
- Can view later

### 6. Comprehensive Logging
Detailed console output:
- Track progress
- Debug issues
- Understand results

---

## Extension Points

### Adding New Models

1. Define model class in `models.py`
2. Add training logic in `train_centralized.py` or use existing
3. Update `main.py` to include new model
4. Update `create_models()` function

### Adding New Datasets

1. Create new loading function in `data_loader.py`
2. Ensure output format matches existing
3. Update `load_and_combine_datasets()`

### Adding New Metrics

1. Add metric calculation in `evaluate_model()`
2. Update `print_comparison_table()` to display
3. Add visualization in `visualize.py` if needed

### Adding New Training Modes

1. Create new training function
2. Add command-line argument in `main.py`
3. Update flow in `main()`

---

## Testing

### Manual Testing

```bash
# Test centralized training
python main.py --train-centralized

# Test federated training
python main.py --train-federated

# Test both
python main.py --train-centralized --train-federated
```

### Verify Outputs

```bash
# Check models saved
ls -l saved_models/

# Check plots generated
ls -l plots/

# Check console output
python main.py 2>&1 | tee training.log
```

---

## Maintenance

### Regular Updates

1. Update `requirements.txt` for new dependencies
2. Update documentation for new features
3. Version control for models
4. Archive old plots

### Performance Monitoring

1. Track training times
2. Monitor memory usage
3. Compare model performance
4. Review generated plots

---

## Best Practices

### When Running

1. Check available disk space
2. Monitor GPU memory if using CUDA
3. Save console output for records
4. Backup trained models

### When Modifying

1. Test changes on small dataset first
2. Update documentation
3. Check all imports
4. Verify backward compatibility

### When Deploying

1. Use saved models, not training scripts
2. Include scaler.pkl for preprocessing
3. Document model version
4. Test on validation data first

---

## Summary

This IDS Detection System provides a complete, modular, and well-documented framework for training and evaluating intrusion detection models using both centralized and federated learning approaches. All code is organized, documented, and ready for production use.

