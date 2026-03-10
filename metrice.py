"""
Model Evaluation Script for ZTF Eclipsing Binary Classification

This module provides comprehensive evaluation metrics including:
- Classification reports (precision, recall, F1-score)
- Confusion matrices
- Model complexity analysis (FLOPs, parameters)

Usage:
    python metrice.py

Output:
    - Classification reports printed to console
    - Confusion matrix visualizations saved as PNG files

Author: [Your Name]
Date: 2023
"""

import os
import warnings
import itertools

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from PIL import Image
from thop import clever_format, profile

warnings.filterwarnings('ignore')


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion Matrix',
                         cmap=plt.cm.Blues, model_name='model', save_path=None):
    """
    Plot and save confusion matrix visualization.
    
    Args:
        cm: Confusion matrix (numpy array)
        classes: List of class names
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        cmap: Colormap for visualization
        model_name: Name of the model (for filename)
        save_path: Path to save the figure
        
    Returns:
        Normalized confusion matrix
    """
    plt.figure(figsize=(10, 10))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f'{model_name} - {title}', fontsize=18)
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i,
            f'{cm[i, j]:.2f}',
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=12
        )
    
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    
    if save_path is None:
        save_path = f'{model_name}_cm.png'
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return cm


# Preprocessing transform for inference
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])


def load_class_labels(label_file='classes.txt'):
    """
    Load class labels from text file.
    
    Args:
        label_file: Path to label definition file
        
    Returns:
        List of class names
    """
    with open(label_file, encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def evaluate_model(model, test_dir, device, class_labels):
    """
    Evaluate model performance on test dataset.
    
    Args:
        model: Trained PyTorch model
        test_dir: Directory containing test images
        device: Computation device
        class_labels: List of class names
        
    Returns:
        Tuple of (y_true, y_pred, accuracy)
    """
    model.eval()
    y_pred, y_true = [], []
    
    for class_idx in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_idx)
        if not os.path.isdir(class_path):
            continue
            
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img_tensor = INFERENCE_TRANSFORM(img)
                img_tensor = torch.unsqueeze(img_tensor, dim=0).to(device).float()
                
                # Inference
                with torch.no_grad():
                    output = model(img_tensor)
                    pred = np.argmax(output.cpu().detach().numpy()[0])
                
                y_pred.append(pred)
                y_true.append(int(class_idx))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    accuracy = accuracy_score(y_true, y_pred)
    
    return y_true, y_pred, accuracy


def analyze_model_complexity(model, device):
    """
    Calculate model complexity metrics (FLOPs and parameters).
    
    Args:
        model: PyTorch model
        device: Computation device
        
    Returns:
        Tuple of (flops, params) as formatted strings
    """
    dummy_input = torch.randn(1, 1, 224, 224).to(device)
    flops, params = profile(model.to(device), (dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params


def main():
    """
    Main evaluation routine for all trained models.
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}\n")
    
    # Load class labels
    class_labels = load_class_labels('classes.txt')
    print(f"Classes: {class_labels}\n")
    
    # Models to evaluate
    model_names = ['ghostnet', 'mobilenetv2']
    
    for model_name in model_names:
        model_path = f'{model_name}.pt'
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue
        
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name.upper()}")
        print(f"{'='*50}")
        
        # Load model
        model = torch.load(model_path, map_location=device).to(device)
        
        # Model complexity analysis
        flops, params = analyze_model_complexity(model, device)
        print(f"\nModel Complexity:")
        print(f"  FLOPs:  {flops}")
        print(f"  Params: {params}")
        
        # Evaluate on test set
        y_true, y_pred, accuracy = evaluate_model(model, 'test', device, class_labels)
        
        print(f"\nOverall Accuracy: {accuracy:.4f}\n")
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_labels))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, classes=class_labels, model_name=model_name)
        print(f"\nConfusion matrix saved to: {model_name}_cm.png")


if __name__ == '__main__':
    main()
