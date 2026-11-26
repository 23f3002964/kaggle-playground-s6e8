#!/bin/bash
# Run DV-05 TabM/RealMLP on Colab via CLI
# Prerequisites: colab CLI installed (pip install colab-cli)
# Usage: ./run_dv05_colab.sh

set -e

SESSION="s6e8-dv05-tabm"
NOTEBOOK="notebooks/dv05_tabm_colab.ipynb"

echo "=== DV-05 TabM/RealMLP on Colab ==="
echo "Session: $SESSION"
echo "Notebook: $NOTEBOOK"
echo ""

# Check if colab CLI is available
if ! command -v colab &> /dev/null; then
    echo "Installing colab-cli..."
    pip install colab-cli
fi

# Create Colab session with T4 GPU
echo "Creating Colab session with T4 GPU..."
colab new -s "$SESSION" --gpu T4

# Upload data files
echo "Uploading data files..."
colab upload -s "$SESSION" data/train.csv /content/train.csv
colab upload -s "$SESSION" data/test.csv /content/test.csv
colab upload -s "$SESSION" data/sample_submission.csv /content/sample_submission.csv

# Upload notebook
echo "Uploading notebook..."
colab upload -s "$SESSION" "$NOTEBOOK" /content/dv05_tabm_colab.ipynb

# Install requirements
echo "Installing requirements..."
colab install -s "$SESSION" -r requirements.txt
# Also install pytabkit
colab exec -s "$SESSION" -f - << 'EOF'
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytabkit"])
print("pytabkit installed")
EOF

# Execute notebook (this will take 30-45 min)
echo "Executing notebook (30-45 min)..."
echo "Progress will show as [DV05T] lines - grep for them in output"
# Convert notebook to python script and execute
colab exec -s "$SESSION" -f - << 'EOF'
import nbformat
from nbconvert import PythonExporter

# Read notebook
with open('/content/dv05_tabm_colab.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Convert to python
exporter = PythonExporter()
source, _ = exporter.from_notebook_node(nb)

# Execute
exec(source)
EOF

# Download results
echo "Downloading results..."
colab download -s "$SESSION" /content/submission_dv05_*.csv submissions/
colab download -s "$SESSION" /content/dv05_*_oof.csv logs/

# Stop session
echo "Stopping Colab session..."
colab stop -s "$SESSION"

echo "=== Done! Check submissions/ and logs/ ==="
