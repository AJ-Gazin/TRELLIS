#!/bin/bash
# TRELLIS Model Download Script - Standalone version
# This script downloads TRELLIS models with proper authentication

set -e

echo "🤖 TRELLIS Model Downloader"
echo "=============================="

# Check if we're in the right directory
if [ ! -f "example_local_edit.py" ]; then
    echo "❌ Error: Run this script from the TRELLIS directory"
    exit 1
fi

# Initialize conda if needed
if [ -d "/workspace/miniconda3" ]; then
    export PATH="/workspace/miniconda3/bin:$PATH"
    eval "$(/workspace/miniconda3/bin/conda shell.bash hook)"
elif [ -d "$HOME/miniconda3" ]; then
    eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
else
    echo "❌ Error: conda not found. Please install conda first or run quick_setup.sh"
    exit 1
fi

# Activate trellis environment
if conda env list | grep -q "^trellis "; then
    conda activate trellis
else
    echo "❌ Error: 'trellis' environment not found. Please run quick_setup.sh first"
    exit 1
fi

# Check authentication
echo "🔐 Checking Hugging Face authentication..."
if ! huggingface-cli whoami &>/dev/null; then
    echo "❌ Not authenticated with Hugging Face"
    echo "Please login first:"
    echo "  1. Get a token from https://huggingface.co/settings/tokens"
    echo "  2. Run: huggingface-cli login --token YOUR_TOKEN"
    echo "  3. Or set: export HF_TOKEN=YOUR_TOKEN"
    exit 1
fi

USER_INFO=$(huggingface-cli whoami)
echo "✅ Authenticated as: $USER_INFO"

# Run the Python download script
echo "📥 Starting model download..."
python download_models.py $@

echo "✅ Model download complete!"
echo ""
echo "You can now run: python example_local_edit.py"