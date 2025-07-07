#!/bin/bash
# TRELLIS Quick Setup - Complete Installation Script
# Tested July 2025 - One command setup for local editing

set -e

echo "🔧 TRELLIS Local Editing - Quick Setup"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "example_local_edit.py" ]; then
    echo "❌ Error: Run this script from the TRELLIS directory"
    exit 1
fi

# Initialize conda - check persistent location first
if [ -d "/workspace/miniconda3" ]; then
    echo "✅ Found conda in persistent storage"
    export PATH="/workspace/miniconda3/bin:$PATH"
    eval "$(/workspace/miniconda3/bin/conda shell.bash hook)"
elif [ -d "$HOME/miniconda3" ]; then
    echo "📦 Found conda in home directory"
    eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
else
    echo "📦 Installing Miniconda to persistent storage..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /workspace/miniconda3
    rm /tmp/miniconda.sh
    export PATH="/workspace/miniconda3/bin:$PATH"
    eval "$(/workspace/miniconda3/bin/conda shell.bash hook)"
fi

# Create environment if it doesn't exist
if conda env list | grep -q "^trellis "; then
    echo "✅ Found existing 'trellis' environment"
else
    echo "🐍 Creating conda environment 'trellis'..."
    conda create -n trellis python=3.10 -y
fi
conda activate trellis

echo "📋 Installing PyTorch 2.6.0 + CUDA 12.6..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

echo "📦 Installing core dependencies..."
pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d transformers igraph xatlas

echo "🔧 Installing mesh processing libraries..."
pip install pymeshlab pymeshfix pyvista plyfile

echo "⚡ Installing attention backend..."
pip install flash-attn

echo "🎯 Installing sparse convolution..."
pip install spconv-cu126

echo "🎨 Installing rendering dependencies..."
pip install git+https://github.com/NVlabs/nvdiffrast.git
# Install diff-gaussian-rasterization from mip-splatting for kernel_size support
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting
pip install /tmp/mip-splatting/submodules/diff-gaussian-rasterization/
rm -rf /tmp/mip-splatting
pip install git+https://github.com/camenduru/simple-knn.git@main

echo "🔄 Installing additional rendering..."
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast
pip install /tmp/diffoctreerast
rm -rf /tmp/diffoctreerast

echo "🧰 Installing utils3d..."
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Note: Git submodules have been converted to regular files for stability
echo "📁 All dependencies are now included as regular files..."

# Download models using separate script
echo "🤖 Downloading TRELLIS models..."
python download_models.py

# Verify installation
echo "🔍 Verifying installation..."
python -c "
import os
os.environ['ATTN_BACKEND'] = 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'

try:
    import torch
    from trellis import models
    from trellis.pipelines import TrellisImageTo3DPipeline
    import diff_gaussian_rasterization
    import simple_knn
    import diffoctreerast
    print('✅ All imports successful!')
    print(f'✅ PyTorch {torch.__version__} with CUDA {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ Verification failed: {e}')
    exit(1)
"

echo ""
echo "🎉 Setup Complete!"
echo ""
echo "To activate TRELLIS environment in future sessions:"
echo "  source /workspace/mesh_edit_project/TRELLIS/activate_trellis.sh"
echo ""
echo "Or manually:"
echo "  conda activate trellis"
echo "  export ATTN_BACKEND=flash-attn"
echo "  export SPCONV_ALGO=native" 
echo "  python example_local_edit.py"
echo ""
echo "Results will be saved to: results/local_edit/"