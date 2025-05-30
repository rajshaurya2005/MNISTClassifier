# MNIST Image Classifier

A PyTorch implementation of a Convolutional Neural Network for classifying handwritten digits from the MNIST dataset.

## Features

- CNN architecture with 2 convolutional layers and 2 fully connected layers
- Data preprocessing with normalization
- Training with validation split (80/20)
- Dropout regularization to prevent overfitting
- Comprehensive evaluation with accuracy metrics and classification report

## Model Architecture

- **Input**: 28x28 grayscale images
- **Conv Layer 1**: 1→32 channels, 3x3 kernel
- **Conv Layer 2**: 32→64 channels, 3x3 kernel
- **Max Pooling**: 2x2
- **Dropout**: 25% and 50% rates
- **FC Layer 1**: 9216→128 neurons
- **FC Layer 2**: 128→10 neurons (output classes)
- **Activation**: ReLU for hidden layers, Log Softmax for output

## Installation

1. Clone or download the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the classifier:
```bash
python mnist_classifier.py
```

The script will:
1. Download the MNIST dataset automatically
2. Train the model for 10 epochs
3. Display training progress with validation metrics
4. Evaluate on test set and show final accuracy

## Expected Performance

The model typically achieves:
- Training accuracy: ~99%
- Validation accuracy: ~98-99%
- Test accuracy: ~98-99%

## Dataset

The MNIST dataset contains 70,000 images of handwritten digits (0-9):
- 60,000 training images
- 10,000 test images
- Each image is 28x28 pixels in grayscale

## Hardware Requirements

- CPU: Any modern processor
- Memory: 4GB RAM minimum
- GPU: Optional (CUDA-compatible for faster training)
- Storage: ~100MB for dataset and model
