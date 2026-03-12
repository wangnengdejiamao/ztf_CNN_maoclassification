# ZTF Eclipsing Binary Classification - Complete Tutorial

This tutorial walks through the complete workflow of using the ZTF Eclipsing Binary Classification system, from data acquisition to model inference.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Demo](#quick-demo)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Understanding the Output](#understanding-the-output)
6. [Advanced Usage](#advanced-usage)

---

## Prerequisites

- Python 3.8 or higher
- Git
- ~2GB free disk space (for models and sample data)
- Internet connection (for downloading ZTF data)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/wangnengdejiamao/ztf_CNN_maoclassification.git
cd ztf_CNN_maoclassification
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n ztf-classifier python=3.9
conda activate ztf-classifier
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Demo

The fastest way to get started is using the interactive demo script:

```bash
python demo.py
```

This will:
1. Download ZTF data for coordinates **RA=58.470417, DEC=43.256989**
2. Process the light curve
3. Run classification
4. Display results

### Demo Output Example

```
============================================================
  ZTF Eclipsing Binary Classification - Demo
============================================================

Demo Coordinates:
  RA:  58.470417 degrees
  DEC: 43.256989 degrees

[1/4] Downloading ZTF light curve data...
  ✓ Downloaded to: demo_output/ZTFJ...csv

[2/4] Processing light curve data...
  ✓ Light curve plot saved
  ✓ CNN input image saved

[3/4] Loading model...
  ✓ Model loaded successfully

[4/4] Performing classification...

============================================================
PREDICTION RESULTS
============================================================
Coordinates: RA=58.470417, DEC=43.256989

Class Probabilities:
  EA (Algol-type)     :  XX.XX%  
  EW (W UMa-type)     :  XX.XX%  >>> PREDICTION <<<
  Non-EB              :  XX.XX%

FINAL PREDICTION: EW (W UMa-type)
============================================================
```

---

## Step-by-Step Guide

### Step 1: Download ZTF Light Curve Data

The `data_download.py` script downloads light curves from IRSA using multi-threading:

```python
# Edit data_download.py to set your catalog
File_Path = "my_catalog"   # CSV with 'ra' and 'dec' columns
File_Path2 = "output"      # Output directory

# Run the download
python data_download.py
```

**Input Catalog Format (CSV):**
```csv
ra,dec
58.470417,43.256989
123.456789,67.890123
...
```

### Step 2: Convert Light Curves to Images

Light curves must be converted to 224×224 grayscale images for the CNN. This involves:

1. **Period Finding**: Use Lomb-Scargle periodogram
2. **Phase Folding**: Fold the light curve by the period
3. **Image Generation**: Create 2D grayscale representation

```python
# Example conversion code
import matplotlib.pyplot as plt
import numpy as np

def lightcurve_to_image(mjd, mag, output_path):
    """Convert light curve to CNN input image."""
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_subplot(111)
    
    # Normalize
    t_norm = (mjd - mjd.min()) / (mjd.max() - mjd.min())
    m_norm = (mag - mag.min()) / (mag.max() - mag.min())
    
    # Plot
    ax.scatter(t_norm, m_norm, c='black', s=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.savefig(output_path, dpi=100, bbox_inches='tight', 
                pad_inches=0, facecolor='white')
    plt.close()
```

### Step 3: Prepare Dataset

Organize images into class folders:

```
train/
├── EA/          # Algol-type binaries
├── EW/          # W UMa-type binaries
└── Non-EB/      # Non-eclipsing binaries
```

Then split into train/val/test:

```bash
python processing.py \
    --data_path train \
    --label_path classes.txt \
    --val_size 0.1 \
    --test_size 0.2
```

### Step 4: Train Model

```bash
python main.py
```

This trains both GhostNet and MobileNetV2 models. Training will:
- Run for 200 epochs
- Save the best model based on validation accuracy
- Generate training logs (`ghostnet.log`, `mobilenetv2.log`)

**Monitor Training:**

```bash
# View live log
tail -f ghostnet.log

# Plot training curves
python plot_curve.py
```

### Step 5: Evaluate Model

```bash
python metrice.py
```

This generates:
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Per-class performance metrics

### Step 6: Batch Inference

For classifying large datasets:

```python
# Edit predict.py to set your input directories
file_types = ["gperiod", "rperiod"]
file_numbers = ["918", "919", "920"]
base_path = "path/to/your/data"

# Run prediction
python predict.py
```

Output: CSV files with `image_path,predicted_class` format.

---

## Understanding the Output

### Classification Classes

| Class | Description | Physical Meaning |
|-------|-------------|------------------|
| **EA** | Algol-type | Detached binary with distinct, different-depth minima |
| **EW** | W UMa-type | Contact binary with continuous variation |
| **Non-EB** | Non-eclipsing | Insufficient periodic signal |

### Interpreting Confidence Scores

- **>80%**: High confidence prediction
- **50-80%**: Moderate confidence - consider manual inspection
- **<50%**: Low confidence - possible unusual morphology or noisy data

### Confusion Matrix

The confusion matrix shows:
- **Rows**: True classes
- **Columns**: Predicted classes
- **Diagonal**: Correct classifications
- **Off-diagonal**: Misclassifications

---

## Advanced Usage

### Custom Coordinates Demo

```bash
# Run demo on custom coordinates
python demo.py \
    --ra 123.456 \
    --dec 67.890 \
    --model ghostnet.pt \
    --radius 0.0014 \
    --output ./my_results
```

### Fine-tuning on New Data

```python
import torch
import timm

# Load pretrained model
model = timm.create_model('mobilenetv2_100', pretrained=True, 
                          num_classes=3, in_chans=1)

# Load your weights
checkpoint = torch.load('mobilenetv2.pt')
model.load_state_dict(checkpoint.state_dict())

# Fine-tune on new data
# ... training code ...
```

### API Integration

Example of using the model in a web API:

```python
from flask import Flask, request, jsonify
import torch
from PIL import Image
import io

app = Flask(__name__)
model = torch.load('mobilenetv2.pt', map_location='cpu')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))
    # ... preprocessing and inference ...
    return jsonify({'class': predicted_class, 'confidence': confidence})
```

---

## Troubleshooting

### Common Issues

1. **"No ZTF data found"**
   - Check coordinates are within ZTF survey coverage
   - Increase search radius with `--radius 0.0014`
   - Verify internet connection

2. **"Model file not found"**
   - Ensure `mobilenetv2.pt` or `ghostnet.pt` exists
   - Download from releases if needed

3. **CUDA out of memory**
   - Reduce batch size in `main.py`
   - Use CPU inference: `device = torch.device('cpu')`

4. **IRSA download timeout**
   - Check network connectivity
   - Reduce `N_THREADS` in `data_download.py`

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ztf_eb_classification_2023,
  author = {Your Name},
  title = {ZTF Eclipsing Binary Classification with Deep Learning},
  year = {2023},
  url = {https://github.com/wangnengdejiamao/ztf_CNN_maoclassification}
}
```

---

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to the README.md for additional documentation
