#!/usr/bin/env python3
"""
Script to download data from Google Drive and extract it to the data/ folder.
"""

import os
import sys
import zipfile
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

def download_from_google_drive(file_id, output_path):
    """Download file from Google Drive using file ID."""
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading from Google Drive...")
        print(f"URL: {url}")
        print(f"Output: {output_path}")
        
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def extract_if_zip(file_path, extract_to):
    """Extract zip file if the downloaded file is a zip archive."""
    if file_path.suffix.lower() in ['.zip']:
        print(f"Extracting {file_path} to {extract_to}")
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print("Extraction completed successfully!")
            
            # Optionally remove the zip file after extraction
            response = input("Remove the zip file after extraction? (y/n): ").lower()
            if response == 'y':
                file_path.unlink()
                print("Zip file removed.")
            
            return True
        except zipfile.BadZipFile:
            print("Downloaded file is not a valid zip file.")
            return False
        except Exception as e:
            print(f"Error extracting file: {e}")
            return False
    else:
        print("Downloaded file is not a zip archive. Keeping as is.")
        return True

def main():
    # Google Drive file ID extracted from the URL
    # https://drive.google.com/file/d/1lhAaeQCmk2y440PmagA0KmIVBIysVMwu/view?usp=drive_link
    file_id = "1lhAaeQCmk2y440PmagA0KmIVBIysVMwu"
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print(f"Created/verified data directory: {data_dir.absolute()}")
    
    # Install gdown if needed
    if not install_gdown():
        return 1
    
    # Determine output filename (we'll detect the actual filename from the download)
    temp_output = data_dir / "downloaded_file"
    
    # Download the file
    print("Starting download...")
    if download_from_google_drive(file_id, str(temp_output)):
        print("Download completed successfully!")
        
        # Check if it's a zip file and extract if needed
        if temp_output.exists():
            # Try to determine the actual file type
            actual_file = temp_output
            
            # If it's a zip file, extract it
            extract_if_zip(actual_file, data_dir)
            
            print(f"\nData has been downloaded to: {data_dir.absolute()}")
            print("Contents of data directory:")
            for item in data_dir.iterdir():
                print(f"  {item.name}")
                
        else:
            print("Download completed but file not found at expected location.")
            return 1
            
    else:
        print("Download failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
