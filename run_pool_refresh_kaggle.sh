#!/bin/bash
# Run Pool Refresh, Prune & Inject on Kaggle Kernel
# Prerequisites: kaggle CLI configured, pool datasets added

set -e

KERNEL_DIR="notebooks/pool_refresh_prune_inject_kernel"
NOTEBOOK="notebooks/pool_refresh_prune_inject.ipynb"

echo "=== Pool141 Refresh, Prune & Inject - Kaggle Kernel ==="
echo "Kernel dir: $KERNEL_DIR"
echo ""

# Create kernel directory
mkdir -p "$KERNEL_DIR"

# Copy notebook
cp "$NOTEBOOK" "$KERNEL_DIR/kernel.ipynb"

# Create kernel-metadata.json
cat > "$KERNEL_DIR/kernel-metadata.json" << 'KERNEL_EOF'
{
  "id": "adi/pool-refresh-prune-inject",
  "title": "Pool141 Refresh, Prune & Inject Own-Track",
  "code_file": "kernel.ipynb",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": [
    "szymonkapiski/s6e8-oof-library-47-models",
    "najiama/predicting-smartphone-addiction-oof-submission-csv",
    "omidbaghchehsaraei/tabm-for-predicting-smartphone-addiction",
    "donmarch14/s6e8-catboost",
    "ravi20076/playgrounds6e8-public-l2stack-v1",
    "ryota517/s6e8-generator-forensics-leak-safe-cv",
    "factualexplorer/baseline-lgbm-xgb-cb-rank-averaged-oof-tuned",
    "mohankrishnathalla/s6e8-catboost-mlp-tuner-oof-saver",
    "evgendvorkin/s6e8-single-lgb"
  ],
  "competition_sources": ["playground-series-s6e8"]
}
KERNEL_EOF

echo "Pushing kernel..."
kaggle kernels push -p "$KERNEL_DIR"

echo ""
echo "Kernel pushed! Now:"
echo "1. Go to kaggle.com/code/adi/pool-refresh-prune-inject"
echo "2. Run the kernel (CPU, ~10 min)"
echo "3. Download the 5 submission files from output"
echo "4. Submit top 2-3 to public LB (suggested: 1%, 2%, 5% injection)"
echo ""
echo "This will refresh pool141 (2 weeks stale) and test own-track injection"