"""
Configuration file for IDS Detection System
Contains all hyperparameters and settings for training
"""

import os

# Paths
DATA_DIR = '/home/madhu/yuks/MFEDK_IDS/data/CIC-IDS 2017'
SAVE_DIR = '/home/madhu/yuks/MFEDK_IDS/saved_models'
PLOTS_DIR = '/home/madhu/yuks/MFEDK_IDS/plots'

# Create directories if they don't exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Data files
DATA_FILES = {
    'Monday': 'Monday-WorkingHours.pcap_ISCX.csv',
    'Tuesday': 'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday': 'Wednesday-workingHours.pcap_ISCX.csv',
    'Thursday_Morning': 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Thursday_Afternoon': 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'Friday_Morning': 'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Friday_Afternoon_DDoS': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'Friday_Afternoon_PortScan': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
}

# Data preprocessing
BALANCE_SAMPLES = 50000  # Samples per class after balancing
TEST_SIZE = 0.2
RANDOM_STATE = 42
VARIANCE_THRESHOLD = 0.0001

# Training hyperparameters - Centralized
CENTRALIZED_EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# Training hyperparameters - Federated
COMM_ROUND = 10  # Communication rounds
LOCAL_EPOCHS = 1  # Local epochs per round
FED_LEARNING_RATE = 0.001
CLIENT_NUM_IN_TOTAL = 5  # Total number of clients
CLIENT_NUM_PER_ROUND = 5  # Clients participating per round

# Model hyperparameters
AUTOENCODER_BOTTLENECK_DIM = 32
AUTOENCODER_ALPHA = 0.7  # Weight for reconstruction loss

# Device
DEVICE = 'cuda'  # Will be set to 'cpu' if cuda not available

# Random seeds
import numpy as np
import torch

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)

