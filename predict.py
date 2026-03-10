"""
Batch Inference Script for ZTF Eclipsing Binary Classification

This module provides functionality for applying trained models to new
unlabeled datasets, generating predictions for large-scale sky surveys.

Usage:
    Configure the input paths in the main() function, then run:
    python predict.py

Output:
    CSV files containing image paths and predicted class labels

Author: [Your Name]
Date: 2023
"""

import os
import warnings

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import tqdm

# Suppress duplicate library warnings on certain platforms
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings('ignore')


# Inference preprocessing
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])


def load_model(model_path, device):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint (.pt file)
        device: Computation device
        
    Returns:
        Loaded PyTorch model in evaluation mode
    """
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def predict_image(model, image, device):
    """
    Perform inference on a single image.
    
    Args:
        model: Trained PyTorch model
        image: PIL Image object
        device: Computation device
        
    Returns:
        Predicted class index
    """
    # Preprocess
    img_tensor = INFERENCE_TRANSFORM(image)
    img_tensor = torch.unsqueeze(img_tensor, dim=0).to(device).float()
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        prediction = np.argmax(output.cpu().detach().numpy()[0])
    
    return prediction


def batch_predict(model, input_dir, class_labels, device):
    """
    Perform batch prediction on all images in a directory.
    
    Args:
        model: Trained PyTorch model
        input_dir: Directory containing input images
        class_labels: List of class names
        device: Computation device
        
    Returns:
        List of prediction results (filename, predicted_class)
    """
    results = []
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    for img_file in tqdm.tqdm(image_files, desc=f"Processing {os.path.basename(input_dir)}"):
        img_path = os.path.join(input_dir, img_file)
        
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Predict
            pred_idx = predict_image(model, img, device)
            pred_label = class_labels[int(pred_idx)]
            
            results.append(f'{img_file},{pred_label}')
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    return results


def save_predictions(results, output_file):
    """
    Save prediction results to CSV file.
    
    Args:
        results: List of prediction strings
        output_file: Path to output CSV file
    """
    with open(output_file, 'w') as f:
        f.write('image_path,predicted_class\n')
        f.write('\n'.join(results))
    print(f"Predictions saved to: {output_file}")


def load_class_labels(label_file='classes.txt'):
    """
    Load class label definitions.
    
    Args:
        label_file: Path to label file
        
    Returns:
        List of class names
    """
    with open(label_file, encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def main():
    """
    Main inference routine for batch prediction on new datasets.
    
    Configure the following parameters before running:
    - model_path: Path to trained model
    - input_directories: List of directories to process
    - output_prefix: Prefix for output CSV filenames
    """
    # Configuration
    model_path = 'mobilenetv2.pt'  # or 'ghostnet.pt'
    
    # Example configuration for Swift/XRT data processing
    # Modify these paths according to your data structure
    file_types = ["gperiod", "rperiod"]  # Filter bands
    file_numbers = ["918", "919", "920", "921"]  # Field identifiers
    base_path = "path/to/your/data"  # Base directory for input data
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")
    
    # Load model
    print(f"Loading model: {model_path}")
    model = load_model(model_path, device)
    
    # Load class labels
    class_labels = load_class_labels('classes.txt')
    print(f"Classes: {class_labels}\n")
    
    # Process each data subset
    for file_type in file_types:
        for file_number in file_numbers:
            input_dir = os.path.join(base_path, file_number, file_type)
            
            if not os.path.exists(input_dir):
                print(f"Directory not found: {input_dir}")
                continue
            
            print(f"\nProcessing: {input_dir}")
            
            # Perform batch prediction
            results = batch_predict(model, input_dir, class_labels, device)
            
            # Save results
            output_file = f'swift_{file_number}_{file_type}.csv'
            save_predictions(results, output_file)


if __name__ == '__main__':
    main()
