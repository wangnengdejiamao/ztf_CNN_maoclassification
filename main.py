"""
Deep Learning Training Script for ZTF Eclipsing Binary Classification

This module implements the training pipeline for classifying ZTF light curves
using lightweight CNN architectures (GhostNet and MobileNetV2).

Usage:
    python main.py

The script automatically trains both models and saves the best checkpoints
based on validation accuracy.

Author: [Your Name]
Date: 2023
"""

import os
import time
import datetime
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import timm
from PIL import ImageFile
from sklearn.metrics import confusion_matrix

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')


def calculate_confusion_matrix(y_true, y_pred, num_classes):
    """
    Compute confusion matrix from model predictions.
    
    Args:
        y_true: Ground truth labels (tensor)
        y_pred: Model predictions (tensor)
        num_classes: Number of classification classes
        
    Returns:
        Confusion matrix as numpy array
    """
    y_true_np = y_true.to('cpu').detach().numpy()
    y_pred_np = np.argmax(y_pred.to('cpu').detach().numpy(), axis=1)
    y_true_np = y_true_np.reshape((-1))
    y_pred_np = y_pred_np.reshape((-1))
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(num_classes)))
    return cm


# Data augmentation and normalization transforms
DATA_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.Grayscale(),
        transforms.RandomChoice([
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
        ]),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
}


def create_data_loaders(batch_size, num_workers=8):
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        batch_size: Number of samples per batch
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = torchvision.datasets.ImageFolder(
        'train',
        transform=DATA_TRANSFORMS['train']
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_dataset = torchvision.datasets.ImageFolder(
        'val',
        transform=DATA_TRANSFORMS['test']
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, criterion, device, scaler, amp_enabled):
    """
    Execute one training epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        optimizer: Optimization algorithm
        criterion: Loss function
        device: Computation device (CPU/CUDA)
        scaler: Gradient scaler for mixed precision
        amp_enabled: Whether automatic mixed precision is enabled
        
    Returns:
        Tuple of (average_loss, accuracy, confusion_matrix)
    """
    model.train()
    train_losses = []
    train_cm = np.zeros(shape=(CLASS_NUM, CLASS_NUM))
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).long()
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_losses.append(float(loss.data))
        train_cm += calculate_confusion_matrix(labels, outputs, CLASS_NUM)
    
    avg_loss = np.mean(train_losses)
    accuracy = np.diag(train_cm).sum() / (train_cm.sum() + 1e-7)
    
    return avg_loss, accuracy, train_cm


def validate(model, val_loader, criterion, device, amp_enabled):
    """
    Execute validation pass.
    
    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Computation device (CPU/CUDA)
        amp_enabled: Whether automatic mixed precision is enabled
        
    Returns:
        Tuple of (average_loss, accuracy, confusion_matrix)
    """
    model.eval()
    val_losses = []
    val_cm = np.zeros(shape=(CLASS_NUM, CLASS_NUM))
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(inputs.float())
                loss = criterion(outputs, labels)
            
            val_losses.append(float(loss.data))
            val_cm += calculate_confusion_matrix(labels, outputs, CLASS_NUM)
    
    avg_loss = np.mean(val_losses)
    accuracy = np.diag(val_cm).sum() / (val_cm.sum() + 1e-7)
    
    return avg_loss, accuracy, val_cm


def main():
    """
    Main training loop for both GhostNet and MobileNetV2 models.
    """
    global CLASS_NUM
    
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 200
    CLASS_NUM = len(os.listdir('train'))
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Automatic Mixed Precision (AMP) for faster training on CUDA
    amp_enabled = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(BATCH_SIZE)
    
    # Model configurations
    model_names = ["ghostnet", "mobilenetv2"]
    
    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*50}\n")
        
        # Initialize model with pretrained weights
        model = timm.create_model(
            f'{model_name}_100',
            pretrained=True,
            num_classes=CLASS_NUM,
            in_chans=1
        )
        model.to(device)
        model.name = model_name
        
        # Optimizer: AdamW with weight decay
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=1e-4,
            weight_decay=5e-4
        )
        
        # Learning rate scheduler: Cosine annealing
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            eta_min=1e-6,
            T_max=5
        )
        
        # Loss function: Cross-entropy with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
        
        # Training tracking
        best_acc = 0.0
        
        # Initialize log file
        log_file = f'{model.name}.log'
        with open(log_file, 'w') as f:
            f.write('loss,val_loss,acc,val_acc')
        
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Training started')
        
        for epoch in range(EPOCHS):
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_acc, _ = train_epoch(
                model, train_loader, optimizer, criterion,
                device, scaler, amp_enabled
            )
            
            # Validation phase
            val_loss, val_acc, _ = validate(
                model, val_loader, criterion, device, amp_enabled
            )
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                model.to('cpu')
                torch.save(model, f'{model.name}.pt')
                model.to(device)
            
            # Log metrics
            with open(log_file, 'a') as f:
                f.write(f'\n{train_loss:.5f},{val_loss:.5f},{train_acc:.4f},{val_acc:.4f}')
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(
                f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                f'Epoch {epoch+1:3d}/{EPOCHS} | '
                f'Time: {epoch_time:5.2f}s | '
                f'Train Loss: {train_loss:.5f} | '
                f'Val Loss: {val_loss:.5f} | '
                f'Train Acc: {train_acc:.4f} | '
                f'Val Acc: {val_acc:.4f}'
            )
            
            # Update learning rate
            lr_scheduler.step()
        
        print(f"\nTraining completed. Best validation accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main()
