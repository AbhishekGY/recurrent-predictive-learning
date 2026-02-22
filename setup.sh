#!/bin/bash
# Setup script for RPL Inverted Pendulum project
# Run this when starting a new cloud session

set -e

echo "=== Installing Python dependencies ==="
pip install --quiet numpy torch matplotlib scikit-learn pillow

echo "=== Verifying installation ==="
python -c "
import numpy as np
import torch
import matplotlib
import sklearn
from PIL import Image
print(f'numpy: {np.__version__}')
print(f'torch: {torch.__version__}')
print(f'matplotlib: {matplotlib.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print('All packages installed successfully!')
"

echo ""
echo "=== Setup complete ==="
echo "You can now run:"
echo "  python -m pendulum.control --checkpoint checkpoints_v2/rpl_model_final.pt"
echo "  python -m pendulum.animate --checkpoint checkpoints_v2/rpl_model_final.pt --save output.gif"
echo "  python -m pendulum.evaluate --checkpoint checkpoints_v2/rpl_model_final.pt"
echo "  python -m pendulum.visualize --checkpoint checkpoints_v2/rpl_model_final.pt"
