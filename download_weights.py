#!/usr/bin/env python3
"""
Script to download pretrained model weights from Google Drive.
"""

import os
import sys
import subprocess
from pathlib import Path

def install_gdown():
    """Install gdown if it's not already installed."""
    try:
        import gdown
        return True
    except ImportError:
        print("gdown not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
            return True
        except subprocess.CalledProcessError:
            print("Failed to install gdown. Please install it manually: pip install gdown")
            return False

def download_model_weights():
    """Download model weights from Google Drive."""
    # Google Drive file ID extracted from the URL
    # https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view?usp=drive_link
    file_id = "1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG"
    
    # Create weights directory if it doesn't exist
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    print(f"Created/verified weights directory: {weights_dir.absolute()}")
    
    # Install gdown if needed
    if not install_gdown():
        return False
    
    # Import gdown after installation
    import gdown
    
    # Output path for the model weights
    output_path = weights_dir / "model_best.pt"
    
    # Download the file
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading model weights from Google Drive...")
    print(f"URL: {url}")
    print(f"Output: {output_path}")
    
    try:
        gdown.download(url, str(output_path), quiet=False)
        print(f"\nModel weights downloaded successfully to: {output_path}")
        
        # Check file size
        file_size = output_path.stat().st_size
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error downloading model weights: {e}")
        return False

def main():
    print("Downloading pretrained tennis court detection model weights...")
    
    if download_model_weights():
        print("\n✅ Model weights download completed successfully!")
        print("You can now use the pretrained model for inference:")
        print("  python infer_in_image.py --model_path weights/model_best.pt --input_path your_image.jpg --output_path result.jpg")
    else:
        print("\n❌ Failed to download model weights.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
