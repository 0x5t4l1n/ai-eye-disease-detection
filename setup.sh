#!/bin/bash

# Eye Disease Detection Training Setup Script

echo "=== Eye Disease Detection Training Setup ==="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment (recommended)
echo "Creating virtual environment..."
python3 -m venv eye_disease_env
source eye_disease_env/bin/activate  # On Windows: eye_disease_env\Scripts\activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install tensorflow==2.13.0
pip install opencv-python==4.8.0.76
pip install Pillow==10.0.0
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install kaggle==1.5.16
pip install tqdm==4.66.1

# Setup Kaggle API
echo "Setting up Kaggle API..."
mkdir -p ~/.kaggle

# Check if kaggle.json exists
if [ -f "kaggle.json" ]; then
    cp kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    echo "Kaggle API configured successfully!"
else
    echo "Please ensure kaggle.json is in the current directory"
    echo "Your kaggle.json should contain:"
    echo '{"username":"<Username>","key":"<YourKey>"}'
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p models
mkdir -p logs
mkdir -p temp

# Test Kaggle API
echo "Testing Kaggle API..."
kaggle datasets list --max-size 1 > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Kaggle API working correctly!"
else
    echo "❌ Kaggle API test failed. Please check your credentials."
    exit 1
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To start training, run:"
echo "python training_pipeline.py"
echo ""
echo "The training will:"
echo "1. Download the fundus dataset (may take a few minutes)"
echo "2. Process 1000+ images across 39 categories"
echo "3. Train a deep learning model (may take 1-2 hours)"
echo "4. Save the trained model in the 'models' directory"
echo ""
echo "Note: Training time depends on your hardware. GPU is recommended."
