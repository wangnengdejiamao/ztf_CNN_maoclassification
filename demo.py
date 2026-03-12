"""
ZTF Eclipsing Binary Classification - Interactive Demo

This script demonstrates the complete workflow of the ZTF eclipsing binary
classification system using a specific sky coordinate (RA, DEC).

Example Usage:
    # Run the demo with default coordinates (RA=58.470417, DEC=43.256989)
    python demo.py
    
    # Run with custom coordinates
    python demo.py --ra 58.470417 --dec 43.256989 --model mobilenetv2.pt

Author: [Your Name]
Date: 2023
"""

import os
import sys
import argparse
import warnings
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
import astropy.coordinates as coord
from astropy import units as u
import wget

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Default demo coordinates
DEFAULT_RA = 58.470417
DEFAULT_DEC = 43.256989

# IRSA API endpoint for ZTF light curves
IRSA_API_URL = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"

# Class labels
CLASS_LABELS = ["EA (Algol-type)", "EW (W UMa-type)", "Non-EB"]
CLASS_DESCRIPTIONS = {
    0: "Algol-type (β Persei) eclipsing binary - Detached system with well-defined minima",
    1: "W UMa-type eclipsing binary - Contact or near-contact binary with continuous variation",
    2: "Non-eclipsing binary - Insufficient periodic variation for classification"
}


def parse_arguments():
    """
    Parse command-line arguments for the demo.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='ZTF Eclipsing Binary Classification Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default demo coordinates
  python demo.py
  
  # Run with custom coordinates and specific model
  python demo.py --ra 58.470417 --dec 43.256989 --model ghostnet.pt
  
  # Save results to specific directory
  python demo.py --output ./demo_results
        """
    )
    parser.add_argument(
        '--ra',
        type=float,
        default=DEFAULT_RA,
        help=f'Right Ascension in degrees (default: {DEFAULT_RA})'
    )
    parser.add_argument(
        '--dec',
        type=float,
        default=DEFAULT_DEC,
        help=f'Declination in degrees (default: {DEFAULT_DEC})'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='mobilenetv2.pt',
        choices=['mobilenetv2.pt', 'ghostnet.pt'],
        help='Model checkpoint to use (default: mobilenetv2.pt)'
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=0.00083,
        help='Search radius in degrees (default: 0.00083 = 3 arcsec)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='demo_output',
        help='Output directory for results (default: demo_output)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip generating visualization plots'
    )
    return parser.parse_args()


def create_mock_light_curve(ra, dec, output_dir='demo_output', eb_type='EW'):
    """
    Create a mock ZTF light curve for demonstration purposes.
    
    Args:
        ra: Right Ascension in degrees
        dec: Declination in degrees
        output_dir: Directory to save mock data
        eb_type: 'EW' for W UMa-type, 'EA' for Algol-type
        
    Returns:
        Path to mock CSV file
    """
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create coordinate string for filename
    c = coord.SkyCoord(ra, dec, unit='deg', frame='icrs')
    coord_str = c.to_string('hmsdms', sep='', precision=2).replace(" ", "")
    output_file = os.path.join(output_dir, f'ZTFJ{coord_str}.csv')
    
    # Generate mock eclipsing binary data
    np.random.seed(42)
    
    # Observation times (MJD) - simulate ~2 years of observations
    mjd_start = 58500  # ~2019
    n_obs = 200
    # Random observation times (realistic: not uniform, with gaps)
    mjd = []
    t = mjd_start
    while len(mjd) < n_obs:
        # Typical ZTF cadence: every few days
        t += np.random.exponential(3.0)  # Mean 3 days between obs
        # Only observe at night (random hour)
        if np.random.random() > 0.3:  # 70% chance of good weather
            mjd.append(t)
    mjd = np.array(mjd[:n_obs])
    
    # Period and magnitude based on type
    if eb_type == 'EW':
        # W UMa-type: short period (~0.3-0.5 days), similar depth minima
        period = 0.3542
        mag_base = 15.5
        primary_depth = 0.35
        secondary_depth = 0.25
        # Continuous variation (contact binary)
        continuum = 0.05
    else:  # EA
        # Algol-type: longer period (~1-10 days), different depth minima
        period = 2.867
        mag_base = 14.8
        primary_depth = 0.6
        secondary_depth = 0.15
        # Flat outside eclipse (detached binary)
        continuum = 0.0
    
    # Generate phase
    phase = ((mjd - mjd_start) % period) / period
    
    # Eclipsing binary light curve model
    # Primary eclipse at phase 0.0 (narrower)
    primary_width = 0.06 if eb_type == 'EW' else 0.04
    primary_eclipse = primary_depth * np.exp(-((phase % 1.0) / primary_width) ** 2)
    
    # Secondary eclipse at phase 0.5 (narrower for EA)
    secondary_width = 0.06 if eb_type == 'EW' else 0.03
    secondary_eclipse = secondary_depth * np.exp(-(((phase - 0.5) % 1.0) / secondary_width) ** 2)
    
    # Ellipsoidal variation (only for EW/contact binaries)
    ellipsoidal = continuum * np.cos(2 * np.pi * phase) if eb_type == 'EW' else 0
    
    # Combine
    mag = mag_base + primary_eclipse + secondary_eclipse + ellipsoidal
    
    # Add scatter (photometric noise, magnitude-dependent)
    # Brighter stars have better photometry
    magerr = 0.02 + 0.03 * np.random.exponential(0.5, n_obs)
    mag += np.random.normal(0, magerr * 0.5)
    
    # Create DataFrame with realistic ZTF columns
    df = pd.DataFrame({
        'mjd': mjd,
        'mag': mag,
        'magerr': magerr,
        'filtercode': ['zr'] * n_obs,  # r-band
        'ra': [ra] * n_obs,
        'dec': [dec] * n_obs
    })
    
    df.to_csv(output_file, index=False)
    print(f"  Generated {eb_type}-type eclipsing binary mock data ({n_obs} observations)")
    return output_file


def download_light_curve(ra, dec, radius=0.00083, output_dir='demo_output', use_mock=False):
    """
    Download ZTF light curve data from IRSA for a given sky coordinate.
    
    Args:
        ra: Right Ascension in degrees
        dec: Declination in degrees
        radius: Search radius in degrees
        output_dir: Directory to save downloaded data
        use_mock: If True, create mock data instead of downloading
        
    Returns:
        Path to CSV file (downloaded or mock)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create coordinate string for filename
    c = coord.SkyCoord(ra, dec, unit='deg', frame='icrs')
    coord_str = c.to_string('hmsdms', sep='', precision=2).replace(" ", "")
    output_file = os.path.join(output_dir, f'ZTFJ{coord_str}.csv')
    
    # Use mock data if requested or as fallback
    if use_mock:
        print(f"\n[1/4] Creating mock light curve data...")
        print(f"  (Simulating EW-type eclipsing binary for demonstration)")
        return create_mock_light_curve(ra, dec, output_dir)
    
    # Construct API URL
    api_url = f"{IRSA_API_URL}?POS=CIRCLE+{ra}+{dec}+{radius}&FORMAT=csv"
    
    print(f"\n[1/4] Downloading ZTF light curve data...")
    print(f"  Coordinates: RA={ra}, DEC={dec}")
    print(f"  Search radius: {radius} deg ({radius*3600:.1f} arcsec)")
    
    try:
        wget.download(api_url, out=output_file)
        
        # Check if file was created and has content
        if os.path.exists(output_file) and os.path.getsize(output_file) > 100:
            # Check if it's valid CSV with expected columns
            with open(output_file, 'r') as f:
                content = f.read()
            if 'mjd' in content.lower() or 'mag' in content.lower() or 'ra' in content.lower():
                print(f"\n  ✓ Downloaded to: {output_file}")
                return output_file
        
        # If file is empty or invalid, remove it and use mock
        if os.path.exists(output_file):
            os.remove(output_file)
        
        print(f"\n  ⚠ No ZTF data available for this coordinate")
        print(f"  Switching to mock data mode for demonstration...")
        return create_mock_light_curve(ra, dec, output_dir, eb_type='EW')
        
    except Exception as e:
        print(f"\n  ⚠ Download failed: {e}")
        print(f"  Using mock data for demonstration...")
        return create_mock_light_curve(ra, dec, output_dir, eb_type='EW')


def process_light_curve(csv_file, output_dir='demo_output'):
    """
    Process raw ZTF light curve data and convert to image format.
    
    This function:
    1. Loads the CSV data
    2. Performs phase folding if period information is available
    3. Generates a 2D grayscale image representation
    
    Args:
        csv_file: Path to CSV file containing light curve data
        output_dir: Directory to save processed images
        
    Returns:
        Path to generated image file or None if failed
    """
    print(f"\n[2/4] Processing light curve data...")
    
    try:
        # Load data
        df = pd.read_csv(csv_file)
        
        if len(df) == 0:
            print("  ✗ No data found in the file")
            return None
        
        print(f"  Total observations: {len(df)}")
        
        # Check for required columns
        required_cols = ['mjd', 'mag', 'magerr']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ✗ Missing columns: {missing_cols}")
            return None
        
        # Filter valid measurements
        df = df[df['mag'] > 0].copy()
        
        # Extract data
        mjd = df['mjd'].values
        mag = df['mag'].values
        magerr = df['magerr'].values
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Light curve vs time
        ax1 = axes[0]
        ax1.errorbar(mjd, mag, yerr=magerr, fmt='o', markersize=4, alpha=0.6, color='blue')
        ax1.invert_yaxis()  # Brighter = lower magnitude
        ax1.set_xlabel('MJD (Modified Julian Date)')
        ax1.set_ylabel('Magnitude')
        ax1.set_title('ZTF Light Curve')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Phase-folded light curve (if period available)
        ax2 = axes[1]
        
        # Try to estimate period using simple string length method
        # This is a simplified version - real period finding would use Lomb-Scargle
        try:
            from scipy import signal
            
            # Normalize time and magnitude
            t_norm = (mjd - mjd.min()) / (mjd.max() - mjd.min()) if mjd.max() > mjd.min() else mjd * 0
            m_norm = (mag - mag.mean()) / mag.std() if mag.std() > 0 else mag * 0
            
            # Simple period estimation (this is a placeholder for real period finding)
            period = estimate_period(mjd, mag)
            
            if period > 0:
                phase = ((mjd - mjd.min()) / period) % 1.0
                ax2.scatter(phase, mag, c='red', alpha=0.6, s=20)
                ax2.scatter(phase + 1, mag, c='red', alpha=0.6, s=20)  # Double for visualization
                ax2.invert_yaxis()
                ax2.set_xlabel('Phase')
                ax2.set_ylabel('Magnitude')
                ax2.set_title(f'Phase-folded Light Curve (Period ≈ {period:.4f} days)')
                ax2.set_xlim(0, 2)
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Period estimation failed\n(Using raw light curve)',
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_xticks([])
                ax2.set_yticks([])
        except Exception as e:
            ax2.text(0.5, 0.5, f'Phase folding error: {str(e)}',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=10)
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        plt.tight_layout()
        
        # Save plot
        base_name = os.path.basename(csv_file).replace('.csv', '')
        plot_file = os.path.join(output_dir, f'{base_name}_lightcurve.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Light curve plot saved: {plot_file}")
        
        # Generate CNN input image (224x224 grayscale)
        image_file = generate_cnn_image(mjd, mag, magerr, output_dir, base_name)
        
        plt.close()
        return image_file
        
    except Exception as e:
        print(f"  ✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def estimate_period(mjd, mag):
    """
    Simple period estimation using Lomb-Scargle periodogram.
    
    Args:
        mjd: Modified Julian Dates
        mag: Magnitude values
        
    Returns:
        Estimated period in days
    """
    try:
        from astropy.timeseries import LombScargle
        
        # Create Lomb-Scargle periodogram
        ls = LombScargle(mjd, mag)
        
        # Set frequency range (0.1 to 10 cycles/day, i.e., 0.1 to 10 days period)
        frequency, power = ls.autopower(
            minimum_frequency=0.1,
            maximum_frequency=10.0,
            samples_per_peak=10
        )
        
        # Find peak
        best_frequency = frequency[np.argmax(power)]
        period = 1.0 / best_frequency
        
        return period
    except Exception:
        # Fallback: return median time difference as rough estimate
        if len(mjd) > 1:
            return np.median(np.diff(np.sort(mjd))) * 10
        return 1.0


def generate_cnn_image(mjd, mag, magerr, output_dir, base_name):
    """
    Generate a 224x224 grayscale image suitable for CNN input.
    
    Args:
        mjd: Modified Julian Dates
        mag: Magnitude values
        magerr: Magnitude errors
        output_dir: Output directory
        base_name: Base filename
        
    Returns:
        Path to generated image
    """
    # Create a 224x224 grayscale image
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_subplot(111)
    
    # Normalize magnitude to [0, 1] range
    mag_min, mag_max = mag.min(), mag.max()
    mag_norm = (mag - mag_min) / (mag_max - mag_min + 1e-7)
    
    # Normalize time to [0, 1] range
    t_min, t_max = mjd.min(), mjd.max()
    t_norm = (mjd - t_min) / (t_max - t_min + 1e-7)
    
    # Plot as grayscale scatter
    ax.scatter(t_norm, mag_norm, c='black', s=1, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # Save as grayscale image
    image_file = os.path.join(output_dir, f'{base_name}_cnn_input.png')
    plt.savefig(image_file, dpi=100, bbox_inches='tight', pad_inches=0, 
                facecolor='white', format='png')
    plt.close(fig)
    
    # Convert to grayscale
    img = Image.open(image_file).convert('L')
    img = img.resize((224, 224))
    img.save(image_file)
    
    print(f"  ✓ CNN input image saved: {image_file}")
    return image_file


def load_model(model_path, device):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Computation device
        
    Returns:
        Loaded PyTorch model
    """
    print(f"\n[3/4] Loading model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"  ✗ Model file not found: {model_path}")
        print(f"  Please ensure the model checkpoint exists in the current directory")
        return None
    
    try:
        # Use weights_only=False for compatibility with older model files
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.to(device)
        model.eval()
        print(f"  ✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return None


def predict(image_path, model, device):
    """
    Perform inference on a single image.
    
    Args:
        image_path: Path to input image
        model: Trained PyTorch model
        device: Computation device
        
    Returns:
        Tuple of (predicted_class_index, confidence_scores)
    """
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0).to(device).float()
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probabilities, dim=1).item()
        confidences = probabilities.cpu().numpy()[0]
    
    return pred_idx, confidences


def visualize_prediction(image_path, pred_idx, confidences, output_dir, ra, dec):
    """
    Create visualization of prediction results.
    
    Args:
        image_path: Path to input image
        pred_idx: Predicted class index
        confidences: Confidence scores for each class
        output_dir: Output directory
        ra: Right Ascension
        dec: Declination
        
    Returns:
        Path to visualization image
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel: Input image
    ax1 = axes[0]
    img = Image.open(image_path)
    ax1.imshow(img, cmap='gray')
    ax1.set_title('CNN Input Image (224×224)')
    ax1.axis('off')
    
    # Right panel: Prediction results
    ax2 = axes[1]
    colors = ['#2ecc71' if i == pred_idx else '#95a5a6' for i in range(len(CLASS_LABELS))]
    bars = ax2.barh(CLASS_LABELS, confidences * 100, color=colors, edgecolor='black')
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('Confidence (%)')
    ax2.set_title(f'Classification Results\nRA={ra}, DEC={dec}')
    ax2.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    
    # Add percentage labels
    for bar, conf in zip(bars, confidences):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{conf*100:.1f}%', va='center', fontsize=11)
    
    # Add prediction text
    pred_text = f"Prediction: {CLASS_LABELS[pred_idx]}"
    desc_text = CLASS_DESCRIPTIONS[pred_idx]
    ax2.text(0.5, -0.15, f"{pred_text}\n{desc_text}", 
            transform=ax2.transAxes, ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    viz_file = os.path.join(output_dir, 'prediction_result.png')
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Prediction visualization saved: {viz_file}")
    
    plt.close()
    return viz_file


def print_results(pred_idx, confidences, ra, dec):
    """
    Print formatted prediction results.
    
    Args:
        pred_idx: Predicted class index
        confidences: Confidence scores
        ra: Right Ascension
        dec: Declination
    """
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Coordinates: RA={ra}, DEC={dec}")
    print(f"\nClass Probabilities:")
    for i, (label, conf) in enumerate(zip(CLASS_LABELS, confidences)):
        marker = " >>> PREDICTION <<<" if i == pred_idx else ""
        print(f"  {label:20s}: {conf*100:6.2f}%{marker}")
    
    print(f"\n{'='*60}")
    print(f"FINAL PREDICTION: {CLASS_LABELS[pred_idx]}")
    print(f"{'='*60}")
    print(f"Description: {CLASS_DESCRIPTIONS[pred_idx]}")
    print(f"{'='*60}\n")


def main():
    """
    Main demo routine.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Print banner
    print("\n" + "="*60)
    print("  ZTF Eclipsing Binary Classification - Demo")
    print("="*60)
    print(f"\nDemo Coordinates:")
    print(f"  RA:  {args.ra} degrees")
    print(f"  DEC: {args.dec} degrees")
    print(f"\nModel: {args.model}")
    print(f"Output directory: {args.output}")
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Step 1: Download light curve data (or use mock data)
    csv_file = download_light_curve(args.ra, args.dec, args.radius, args.output, use_mock=False)
    if not csv_file:
        print("\n✗ Demo failed: Could not obtain light curve data")
        return 1
    
    # Step 2: Process light curve
    image_file = process_light_curve(csv_file, args.output)
    if not image_file:
        print("\n✗ Demo failed: Could not process light curve data")
        return 1
    
    # Step 3: Load model
    model = load_model(args.model, device)
    if not model:
        print("\n✗ Demo failed: Could not load model")
        print(f"  Please ensure {args.model} exists in the current directory")
        return 1
    
    # Step 4: Predict
    print(f"\n[4/4] Performing classification...")
    pred_idx, confidences = predict(image_file, model, device)
    
    # Print results
    print_results(pred_idx, confidences, args.ra, args.dec)
    
    # Visualize results
    if not args.no_plot:
        visualize_prediction(image_file, pred_idx, confidences, args.output, args.ra, args.dec)
    
    print(f"✓ Demo completed successfully!")
    print(f"  Results saved in: {args.output}/")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
