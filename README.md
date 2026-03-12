# ZTF Eclipsing Binary Classification with Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of a convolutional neural network (CNN) based classification system for identifying eclipsing binary stars from the Zwicky Transient Facility (ZTF) time-series photometric data. The project addresses the challenge of mining rare astronomical objects from large-scale sky survey datasets.

## Overview

Eclipsing binary stars are valuable astrophysical laboratories for determining stellar parameters and testing stellar evolution models. However, identifying these systems from billions of light curves in modern sky surveys presents significant computational challenges. This work employs deep learning techniques to automate the classification of eclipsing binary light curves, achieving high accuracy on ZTF data.

### Key Features

- **Multi-threaded data acquisition**: Efficient parallel downloading of ZTF light curve data via IRSA API
- **Deep learning classification**: Implementation of lightweight CNN architectures (GhostNet, MobileNetV2) optimized for astronomical time-series analysis
- **Comprehensive evaluation**: Detailed performance metrics including confusion matrices, classification reports, and training curves
- **Production-ready inference**: Batch prediction pipeline for large-scale sky surveys

## Dataset

### Data Source

The light curve data are obtained from the [Zwicky Transient Facility (ZTF)](https://www.ztf.caltech.edu/), a large-scale optical sky survey conducted at the Palomar Observatory. ZTF provides photometric observations in g and r bands, capturing temporal variations of celestial objects across the northern sky.

Data access is facilitated through the [NASA/IPAC Infrared Science Archive (IRSA)](https://irsa.ipac.caltech.edu/).

### Classification Categories

The model classifies light curves into three categories:

| Class | Description | Physical Characteristics |
|-------|-------------|-------------------------|
| **EA** | Algol-type (β Persei) eclipsing binaries | Detached systems with well-defined primary and secondary minima of different depths |
| **EW** | W Ursae Majoris-type eclipsing binaries | Contact or near-contact binaries with continuous light variation and nearly equal minima depths |
| **Useless** | Non-periodic or noisy data | Insufficient photometric quality for classification |

### Data Statistics

| Split | EA (Class 0) | EW (Class 1) | Useless (Class 2) | Total |
|-------|-------------|--------------|-------------------|-------|
| Training | 492 | 1,672 | 5,368 | 7,532 |
| Validation | 61 | 209 | 671 | 941 |
| Test | 124 | 418 | 1,342 | 1,884 |

**Note**: The dataset exhibits class imbalance, with "Useless" samples comprising the majority. This reflects the realistic distribution in sky survey data where genuine eclipsing binaries are rare compared to field stars and noise.

## Methodology

### Data Preprocessing

The raw ZTF light curves are converted into phase-folded, normalized images suitable for CNN input:

1. **Period folding**: Light curves are folded using determined periods to create phase diagrams
2. **Image generation**: Phase-folded light curves are rendered as 2D grayscale images
3. **Normalization**: Pixel values are normalized to [0, 1] range
4. **Augmentation**: Random rotations (up to 45°) and vertical flips are applied during training

### Model Architecture

Two lightweight CNN architectures are implemented to balance computational efficiency and classification accuracy:

#### GhostNet
- **Description**: A novel architecture utilizing Ghost modules to reduce computational cost while maintaining representative feature maps
- **Parameters**: ~5.2M
- **FLOPs**: ~141M
- **Input**: 1×224×224 grayscale images

#### MobileNetV2
- **Description**: Inverted residual blocks with linear bottlenecks, optimized for mobile and edge devices
- **Parameters**: ~3.5M
- **FLOPs**: ~300M
- **Input**: 1×224×224 grayscale images

### Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Initial Learning Rate | 1×10⁻⁴ |
| Weight Decay | 5×10⁻⁴ |
| LR Scheduler | Cosine Annealing (T_max=5, η_min=1×10⁻⁶) |
| Batch Size | 32 |
| Epochs | 200 |
| Loss Function | Cross-Entropy with Label Smoothing (ε=0.1) |
| Mixed Precision | FP16 (when CUDA available) |

## Repository Structure

```
.
├── README.md                 # Project documentation
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
├── classes.txt              # Class label definitions
│
├── data_download.py         # Multi-threaded ZTF data downloader
├── processing.py            # Dataset splitting utility
├── main.py                  # Training script
├── metrice.py               # Evaluation and confusion matrix
├── predict.py               # Batch inference script
├── plot_curve.py            # Training curve visualization
│
├── train/                   # Training dataset (images)
├── val/                     # Validation dataset
├── test/                    # Test dataset
│
├── ghostnet.pt              # Trained GhostNet model
├── mobilenetv2.pt           # Trained MobileNetV2 model
├── ghostnet.log             # GhostNet training log
├── mobilenetv2.log          # MobileNetV2 training log
├── curve.png                # Training curves comparison
├── ghostnet_cm.png          # GhostNet confusion matrix
└── mobilenetv2_cm.png       # MobileNetV2 confusion matrix
```

## Installation

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ztf_CNN_maoclassification.git
cd ztf_CNN_maoclassification

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start Tutorial

### Demo: Classify a Single Source

This tutorial demonstrates the complete workflow using a specific sky coordinate (**RA=58.470417, DEC=43.256989**).

#### Step 1: Run the Demo Script

```bash
# Run with default demo coordinates
python demo.py

# Or specify custom coordinates
python demo.py --ra 58.470417 --dec 43.256989 --model mobilenetv2.pt
```

#### Step 2: Demo Output

The demo will:
1. **Download** ZTF light curve data from IRSA for the specified coordinates
2. **Process** the light curve (phase folding, normalization)
3. **Generate** a 224×224 grayscale image for CNN input
4. **Classify** using the trained model
5. **Display** results with confidence scores

#### Step 3: Example Output

```
============================================================
PREDICTION RESULTS
============================================================
Coordinates: RA=58.470417, DEC=43.256989

Class Probabilities:
  EA (Algol-type)     :  15.23%
  EW (W UMa-type)     :  78.45%  >>> PREDICTION <<<
  Non-EB              :   6.32%

============================================================
FINAL PREDICTION: EW (W UMa-type)
============================================================
Description: Contact or near-contact binary with continuous variation
============================================================
```

#### Demo Files

| File | Description |
|------|-------------|
| `demo.py` | Complete demo script with all functionality |
| `demo_output/` | Output directory containing results |
| `*_lightcurve.png` | Light curve visualization |
| `*_cnn_input.png` | CNN input image (224×224) |
| `prediction_result.png` | Classification results with confidence bars |

#### Demo Options

```bash
python demo.py --help

# Custom coordinates
python demo.py --ra 123.456 --dec 67.890

# Use GhostNet model
python demo.py --model ghostnet.pt

# Larger search radius (5 arcsec)
python demo.py --radius 0.0014

# Custom output directory
python demo.py --output ./my_results
```

---

## Usage

### 1. Data Acquisition

Download ZTF light curves using multi-threaded parallel fetching:

```python
# Configure target coordinates in data_download.py
# Set File_Path (input catalog) and File_Path2 (output directory)

python data_download.py
```

The script utilizes 20 concurrent threads to accelerate downloads from IRSA.

### 2. Dataset Preparation

Split the dataset into training, validation, and test sets:

```bash
python processing.py \
    --data_path train \
    --label_path classes.txt \
    --val_size 0.1 \
    --test_size 0.2
```

### 3. Model Training

Train both GhostNet and MobileNetV2 models:

```bash
python main.py
```

Training logs and best model checkpoints (based on validation accuracy) are automatically saved.

### 4. Model Evaluation

Generate classification reports and confusion matrices:

```bash
python metrice.py
```

This produces per-class precision, recall, F1-score, and normalized confusion matrices.

### 5. Visualization

Plot training curves for loss and accuracy:

```bash
python plot_curve.py
```

### 6. Batch Inference

Apply trained models to new datasets:

```python
# Configure input paths in predict.py
python predict.py
```

Results are saved as CSV files containing image paths and predicted labels.

## Results

### Classification Performance

| Model | Accuracy | EA Precision | EA Recall | EW Precision | EW Recall | Useless Precision | Useless Recall |
|-------|----------|--------------|-----------|--------------|-----------|-------------------|----------------|
| GhostNet | 0.9937 | 0.78 | 0.78 | 0.94 | 0.94 | 0.99 | 0.99 |
| MobileNetV2 | 0.9957 | - | - | - | - | - | - |

### Confusion Matrix (GhostNet)

| True \ Predicted | EA | EW | Useless |
|------------------|-----|-----|---------|
| EA | 0.78 | 0.15 | 0.07 |
| EW | 0.01 | 0.94 | 0.04 |
| Useless | 0.00 | 0.01 | 0.99 |

### Training Convergence

Both models achieve convergence within 200 epochs:
- **GhostNet**: Final validation accuracy ~99.37%
- **MobileNetV2**: Final validation accuracy ~99.57%

Training curves comparing loss and accuracy are available in `curve.png`.

## Scientific Context

### Eclipsing Binaries in Time-Domain Astronomy

Eclipsing binary stars (EBs) provide unique opportunities for:
- **Mass determination**: Direct measurement of stellar masses via Kepler's laws
- **Radius calibration**: Accurate stellar radius measurements through light curve analysis
- **Distance measurement**: Primary distance indicators for nearby galaxies
- **Evolutionary studies**: Testing stellar evolution models across different mass ranges

The EA (Algol) and EW (W UMa) types represent distinct evolutionary stages:
- **EA systems**: Typically detached binaries with periods > 1 day
- **EW systems**: Contact binaries with periods typically < 1 day, representing advanced evolutionary states

### ZTF Survey Characteristics

- **Field of view**: 47 deg² per exposure
- **Cadence**: Nightly coverage of the northern sky
- **Filters**: g (λ_eff = 481 nm) and r (λ_eff = 617 nm)
- **Depth**: r ≈ 20.5 mag (5σ, single epoch)

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@software{ztf_eb_classification_2023,
  author = {Your Name},
  title = {ZTF Eclipsing Binary Classification with Deep Learning},
  year = {2023},
  url = {https://github.com/YOUR_USERNAME/ztf_CNN_maoclassification}
}
```

## Data Acknowledgments

This work makes use of data from:
- The Zwicky Transient Facility (ZTF), funded by the National Science Foundation (NSF) and cooperating institutions
- NASA/IPAC Infrared Science Archive (IRSA), operated by the Jet Propulsion Laboratory, California Institute of Technology

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration inquiries, please open an issue on GitHub or contact the repository maintainer.

---

**Note**: This project was developed as part of research on time-domain astronomical data mining and machine learning applications in large-scale sky surveys.
