#!/bin/bash
# TRELLIS Environment Activation Script
# Source this file to activate the complete TRELLIS environment
# Usage: source /workspace/mesh_edit_project/TRELLIS/activate_trellis.sh

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           TRELLIS Local Editing Environment Setup            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# 1. Setup conda from persistent storage
export PATH="/workspace/miniconda3/bin:$PATH"
eval "$(/workspace/miniconda3/bin/conda shell.bash hook)"

# 2. Verify conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: Conda not found at /workspace/miniconda3"
    echo "Please ensure conda is properly installed in the persistent volume"
    return 1 2>/dev/null || exit 1
fi

# 3. Activate the trellis environment
echo "🔄 Activating conda environment 'trellis'..."
conda activate trellis

# 4. Verify activation
if [[ "$CONDA_DEFAULT_ENV" != "trellis" ]]; then
    echo "❌ Error: Failed to activate trellis environment"
    echo "Run 'conda env list' to check available environments"
    return 1 2>/dev/null || exit 1
fi

# 5. Set required environment variables
export ATTN_BACKEND=flash-attn
export SPCONV_ALGO=native

# 6. Navigate to TRELLIS directory
cd /workspace/mesh_edit_project/TRELLIS

# 7. Quick verification
echo ""
echo "🔍 Verifying installation..."
python -c "
import torch
from trellis import models
print('✅ PyTorch version:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
print('✅ TRELLIS imports successful')
" 2>/dev/null || echo "⚠️  Some imports failed - check installation"

# 8. Display status
echo ""
echo "📊 Environment Status:"
echo "───────────────────────────────────────────────────────────────"
echo "📦 Conda Environment: $CONDA_DEFAULT_ENV"
echo "🐍 Python: $(which python)"
echo "📍 Working Directory: $(pwd)"
echo "🔧 ATTN_BACKEND: $ATTN_BACKEND"
echo "🔧 SPCONV_ALGO: $SPCONV_ALGO"
echo "───────────────────────────────────────────────────────────────"

# 9. Show quick start commands
echo ""
echo "🚀 Quick Start Commands:"
echo "───────────────────────────────────────────────────────────────"
echo "1. Run local editing example:"
echo "   python example_local_edit.py"
echo ""
echo "2. View results:"
echo "   ls results/local_edit/"
echo ""
echo "3. Run quick setup (if needed):"
echo "   ./quick_setup.sh"
echo ""
echo "4. Check documentation:"
echo "   cat CLAUDE.md"
echo "───────────────────────────────────────────────────────────────"
echo ""
echo "✅ TRELLIS environment ready!"