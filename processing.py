"""
Dataset Splitting Utility for ZTF Eclipsing Binary Classification

This module handles the splitting of raw data into training, validation,
and test sets with configurable proportions.

Usage:
    python processing.py --data_path train --val_size 0.1 --test_size 0.2

Output:
    - Creates 'val' and 'test' directories with split data
    - Generates 'classes.txt' with class label definitions

Author: [Your Name]
Date: 2023
"""

import os
import shutil
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(0)


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Split dataset into train/val/test sets'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='train',
        help='Path to raw data directory (organized by class folders)'
    )
    parser.add_argument(
        '--label_path',
        type=str,
        default='classes.txt',
        help='Output path for class label definitions'
    )
    parser.add_argument(
        '--val_size',
        type=float,
        default=0.1,
        help='Proportion of data to use for validation (0-1)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Proportion of data to use for testing (0-1)'
    )
    return parser.parse_args()


def create_class_labels(data_path, label_path):
    """
    Create class label definition file from directory structure.
    
    Args:
        data_path: Path to data directory
        label_path: Output path for label file
        
    Returns:
        List of class names
    """
    class_names = sorted(os.listdir(data_path))
    
    with open(label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names))
    
    print(f"Class labels saved to: {label_path}")
    print(f"Classes: {class_names}")
    
    return class_names


def rename_class_folders(data_path):
    """
    Rename class folders to numeric indices for consistent ordering.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Number of classes
    """
    class_names = sorted(os.listdir(data_path))
    num_classes = len(class_names)
    name_width = len(str(num_classes))
    
    for idx, class_name in enumerate(class_names):
        old_path = os.path.join(data_path, class_name)
        new_path = os.path.join(data_path, str(idx).zfill(name_width))
        os.rename(old_path, new_path)
    
    return num_classes


def split_class_data(class_path, val_size, test_size):
    """
    Split data from a single class into train/val/test sets.
    
    Args:
        class_path: Path to class directory
        val_size: Proportion for validation
        test_size: Proportion for testing
        
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    files = os.listdir(class_path)
    np.random.shuffle(files)
    
    total = len(files)
    test_count = int(total * test_size)
    val_count = int(total * val_size)
    train_count = total - val_count - test_count
    
    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]
    
    return train_files, val_files, test_files


def create_split_directories(base_path):
    """
    Create val and test directories mirroring the structure of train.
    
    Args:
        base_path: Path to train directory
    """
    val_path = base_path.replace('train', 'val')
    test_path = base_path.replace('train', 'test')
    
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    return val_path, test_path


def copy_files(file_list, source_dir, dest_dir):
    """
    Copy files from source to destination directory.
    
    Args:
        file_list: List of filenames to copy
        source_dir: Source directory
        dest_dir: Destination directory
    """
    os.makedirs(dest_dir, exist_ok=True)
    for filename in file_list:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(dest_dir, filename)
        shutil.copy2(src, dst)


def move_files(file_list, source_dir, dest_dir):
    """
    Move files from source to destination directory.
    
    Args:
        file_list: List of filenames to move
        source_dir: Source directory
        dest_dir: Destination directory
    """
    os.makedirs(dest_dir, exist_ok=True)
    for filename in file_list:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(dest_dir, filename)
        shutil.move(src, dst)


def main():
    """
    Main dataset splitting routine.
    """
    args = parse_arguments()
    
    print(f"Dataset Splitting Configuration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Validation size: {args.val_size:.1%}")
    print(f"  Test size: {args.test_size:.1%}")
    print(f"  Training size: {1 - args.val_size - args.test_size:.1%}")
    print()
    
    # Create class labels file
    create_class_labels(args.data_path, args.label_path)
    
    # Rename class folders to numeric indices
    num_classes = rename_class_folders(args.data_path)
    print(f"\nFound {num_classes} classes\n")
    
    # Change to data directory
    os.chdir(args.data_path)
    current_dir = os.getcwd()
    
    # Process each class
    for class_idx in sorted(os.listdir(current_dir)):
        class_path = os.path.join(current_dir, class_idx)
        if not os.path.isdir(class_path):
            continue
        
        print(f"Processing class: {class_idx}")
        
        # Split files
        train_files, val_files, test_files = split_class_data(
            class_path, args.val_size, args.test_size
        )
        
        print(f"  Total: {len(train_files) + len(val_files) + len(test_files)}")
        print(f"  Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        # Copy validation files (keep originals in train)
        val_path = class_path.replace('train', 'val')
        copy_files(val_files, class_path, val_path)
        
        # Move test files (remove from train)
        test_path = class_path.replace('train', 'test')
        move_files(test_files, class_path, test_path)
    
    print("\nDataset splitting completed!")
    print("Train data remains in: train/")
    print("Validation data copied to: val/")
    print("Test data moved to: test/")


if __name__ == '__main__':
    main()
