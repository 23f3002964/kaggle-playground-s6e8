#!/bin/bash
# Run DV-11 Upgrade Caruana HC on Kaggle Kernel

set -e

KERNEL_DIR="notebooks/dv11u_caruana_hc_kernel"
NOTEBOOK="notebooks/dv11u_caruana_hc.ipynb"

echo "=== DV-11 Upgrade: Caruana HC with Replacement + Negative ==="
echo "Kernel dir: $KERNEL_DIR"
echo ""

mkdir -p "$KERNEL_DIR"
cp "$NOTEBOOK" "$KERNEL_DIR/kernel.ipynb"

cat > "$KERNEL_DIR/kernel-metadata.json" << 'KERNEL_EOF'
{
  "id": "adityaraj0612/dv11u-caruana-hc",
  "title": "DV-11U Caruana Hill Climbing Replacement Negative",
  "code_file": "kernel.ipynb",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": [
    "szymonkapiski/s6e8-oof-library-47-models"
  ],
  "competition_sources": ["playground-series-s6e8"]
}
KERNEL_EOF

echo "Pushing kernel..."
kaggle kernels push -p "$KERNEL_DIR"

echo ""
echo "Kernel pushed! Now:"
echo "1. Go to kaggle.com/code/adi/dv11u-caruana-hc"
echo "2. Run the kernel (CPU, ~20 min)"
echo "3. Download the submission file"
echo "4. Submit to public LB"
echo ""
echo "This upgrades DV-11 with Caruana HC (replacement + negative weights)"
echo "Expected gain over simple HC: +0.0001-0.0003"