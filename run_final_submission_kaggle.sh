#!/bin/bash
# Run final submission (fixed) on Kaggle Kernel
# Prerequisites: kaggle CLI configured

set -e

KERNEL_DIR="notebooks/final_submission_fixed_kernel"
NOTEBOOK="notebooks/final_submission_fixed.ipynb"

echo "=== Final Submission Fixed - Kaggle Kernel ==="
echo "Kernel dir: $KERNEL_DIR"
echo ""

# Create kernel directory
mkdir -p "$KERNEL_DIR"

# Copy notebook
cp "$NOTEBOOK" "$KERNEL_DIR/kernel.ipynb"

# Copy data (or use Kaggle dataset)
echo "Copying data..."
cp data/train.csv "$KERNEL_DIR/"
cp data/test.csv "$KERNEL_DIR/"
cp data/sample_submission.csv "$KERNEL_DIR/"

# Create kernel-metadata.json
cat > "$KERNEL_DIR/kernel-metadata.json" << 'KERNEL_EOF'
{
  "id": "adityaraj0612/final-submission-fixed",
  "title": "Final Submission Fixed - XGB+LGB Ensemble",
  "code_file": "kernel.ipynb",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": ["playground-series-s6e8"]
}
KERNEL_EOF

echo "Pushing kernel..."
kaggle kernels push -p "$KERNEL_DIR"

echo ""
echo "Kernel pushed! Now:"
echo "1. Go to kaggle.com/code/adi/final-submission-fixed"
echo "2. Run the kernel"
echo "3. When complete, click 'Submit to Competition' from the output"
echo "4. This will give you the TRUE own-track LB score"
echo ""
echo "Expected CV: ~0.96425"
echo "Previous (buggy) LB: 0.95999 (invalid)"
echo "This submission will reveal the TRUE performance"