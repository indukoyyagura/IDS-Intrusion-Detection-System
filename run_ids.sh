#!/bin/bash
# Quick start script for IDS Detection System

echo "=========================================="
echo "IDS Detection System - Quick Start"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null
then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

echo "Python version:"
python --version
echo ""

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import torch, sklearn, pandas, numpy, matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Some dependencies are missing. Installing..."
    pip install -r requirements.txt
fi
echo ""

# Parse command line arguments
MODE="centralized"
if [ "$1" == "federated" ]; then
    MODE="federated"
elif [ "$1" == "both" ]; then
    MODE="both"
fi

echo "Running in mode: $MODE"
echo ""

# Run the appropriate command
case $MODE in
    centralized)
        echo "Training with centralized learning..."
        python main.py --train-centralized
        ;;
    federated)
        echo "Training with federated learning..."
        python main.py --no-centralized --train-federated
        ;;
    both)
        echo "Training with both centralized and federated learning..."
        python main.py --train-centralized --train-federated
        ;;
esac

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Saved models location: saved_models/"
echo "Generated plots location: plots/"
echo ""
echo "To view results:"
echo "  ls saved_models/"
echo "  ls plots/"

