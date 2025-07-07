#!/usr/bin/env python3
"""
TRELLIS Model Download Script
Downloads TRELLIS models with proper authentication and caching
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import HfHubHTTPError

def check_model_exists(model_id, cache_dir=None):
    """Check if model already exists in cache"""
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    # Convert model ID to cache directory name format
    cache_model_dir = f"models--{model_id.replace('/', '--')}"
    cache_path = Path(cache_dir) / cache_model_dir
    
    if cache_path.exists():
        # Check if it has actual model files (not just metadata)
        model_files = list(cache_path.rglob("*.bin")) + list(cache_path.rglob("*.safetensors")) + list(cache_path.rglob("*.pth"))
        if model_files:
            print(f"✅ Model {model_id} already exists in cache at {cache_path}")
            return True
    
    return False

def download_model_with_auth(model_id, cache_dir=None, force_download=False):
    """Download model with proper authentication handling"""
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    # Check if model already exists and skip if not forcing download
    if not force_download and check_model_exists(model_id, cache_dir):
        print(f"⏭️  Skipping download of {model_id} (already exists)")
        return True
    
    try:
        print(f"📥 Downloading {model_id}...")
        
        # Use snapshot_download with proper authentication
        downloaded_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            resume_download=True,
            force_download=force_download
        )
        
        print(f"✅ Successfully downloaded {model_id} to {downloaded_path}")
        return True
        
    except HfHubHTTPError as e:
        if e.response.status_code == 401:
            print(f"❌ Authentication error for {model_id}")
            print("Please ensure you are logged in to Hugging Face:")
            print("  1. Get a token from https://huggingface.co/settings/tokens")
            print("  2. Run: huggingface-cli login --token YOUR_TOKEN")
            print("  3. Or set: export HF_TOKEN=YOUR_TOKEN")
            return False
        else:
            print(f"❌ HTTP error downloading {model_id}: {e}")
            return False
    except Exception as e:
        print(f"❌ Error downloading {model_id}: {e}")
        return False

def main():
    """Main function to download all required TRELLIS models"""
    print("🤖 TRELLIS Model Downloader")
    print("=" * 40)
    
    # Set environment variables for TRELLIS
    os.environ['ATTN_BACKEND'] = 'flash-attn'
    os.environ['SPCONV_ALGO'] = 'native'
    
    # Check for force download flag
    force_download = "--force" in sys.argv
    if force_download:
        print("🔄 Force download enabled - will re-download existing models")
    
    # List of models to download
    models_to_download = [
        "microsoft/TRELLIS-image-large"
    ]
    
    success_count = 0
    total_count = len(models_to_download)
    
    for model_id in models_to_download:
        if download_model_with_auth(model_id, force_download=force_download):
            success_count += 1
        else:
            print(f"❌ Failed to download {model_id}")
    
    print("\n" + "=" * 40)
    if success_count == total_count:
        print(f"🎉 All {total_count} models downloaded successfully!")
        print("\nModels are cached in: ~/.cache/huggingface/hub/")
        print("\nYou can now run: python example_local_edit.py")
        return 0
    else:
        print(f"❌ Downloaded {success_count}/{total_count} models")
        print("Some models failed to download. Please check authentication and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())