# Remote Execution Capabilities for Playground Series S6E8

## Available Tools
| Tool | Installed | Purpose |
|------|-----------|---------|
| **Colab CLI** | ✅ `/home/adi/.local/bin/colab` | Run code on Google Colab VMs (CPU/GPU/TPU) |
| **Kaggle CLI** | ✅ `/home/adi/.local/bin/kaggle` | Manage Kaggle kernels, submissions, datasets |

---

## Colab CLI — What We Can Do

### 1. **Provision Compute**
```bash
# CPU session (free tier)
colab new -s s6e8-cpu

# GPU session (if quota available)
colab new -s s6e8-gpu --gpu T4      # T4 (usually available)
colab new -s s6e8-gpu --gpu A100    # A100 (premium, may fail)
colab new -s s6e8-tpu --tpu v6e1    # TPU v6e1

# List active sessions
colab sessions
```

### 2. **Run Code Remotely**
```bash
# Run a local script on remote VM (preferred)
colab exec -s s6e8-gpu -f src/train.py

# Run with piped code
cat src/train.py | colab exec -s s6e8-gpu

# Run notebook, save output
colab exec -s s6e8-gpu -f notebooks/01_eda.ipynb

# One-shot: create + run + destroy (no state kept)
colab run --gpu T4 src/train.py
```

### 3. **Environment Setup**
```bash
# Install packages on VM
colab install -s s6e8-gpu -r requirements.txt
colab install -s s6e8-gpu lightgbm xgboost catboost optuna mlflow

# Mount Google Drive (interactive - needs human)
colab drivemount -s s6e8-gpu

# Authenticate VM-side for GCP (BigQuery, GCS)
colab auth -s s6e8-gpu
```

### 4. **File Sync**
```bash
# Upload local file to VM
colab upload -s s6e8-gpu data/train.csv /content/train.csv
colab upload -s s6e8-gpu data/test.csv /content/test.csv

# Download results from VM
colab download -s s6e8-gpu /content/submission.csv submissions/

# List files on VM
colab ls -s s6e8-gpu /content/
```

### 5. **Monitoring & Debugging**
```bash
# Session status (IDLE/BUSY, hardware)
colab status -s s6e8-gpu

# View logs (execution history, keep-alive)
colab log -s s6e8-gpu -n 50

# Export session as notebook
colab log -s s6e8-gpu -o notebooks/session_output.ipynb

# Restart kernel (keep VM)
colab restart-kernel -s s6e8-gpu

# Get browser URL to attach Colab UI
colab url -s s6e8-gpu --open
```

### 6. **Cleanup (CRITICAL - saves compute units)**
```bash
# Stop and release VM
colab stop -s s6e8-gpu

# Stop all
colab sessions | grep -v NAME | awk '{print $1}' | xargs -I {} colab stop -s {}
```

---

## Kaggle CLI — What We Can Do

### 1. **Kaggle Kernels (Notebooks)**
```bash
# Initialize kernel metadata
kaggle kernels init -p notebooks/kaggle_kernel

# Push code and run on Kaggle (free GPU: 30h/week)
# GPU RULE: ALWAYS T4 — kernel-metadata.json me BOTH set karo:
#   "enable_gpu": true  AND  "machine_shape": "NvidiaTeslaT4"
# (enable_gpu akela P100 deta hai; default image ka torch cu128 P100 pe
#  cudaErrorNoKernelImageForDevice se crash hota hai). CLI alt:
#   kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4
kaggle kernels push -p notebooks/kaggle_kernel

# Check status
kaggle kernels status <kernel-slug>

# View logs
kaggle kernels logs <kernel-slug>

# Get output files (models, submissions)
kaggle kernels output <kernel-slug> -p models/
kaggle kernels output <kernel-slug> -p submissions/

# Pull kernel locally
kaggle kernels pull <kernel-slug> -p notebooks/kaggle_kernel
```

### 2. **Competitions**
```bash
# Submit predictions
kaggle competitions submit -c playground-series-s6e8 -f submissions/submission.csv -m "LightGBM baseline"

# List submissions
kaggle competitions submissions -c playground-series-s6e8

# View leaderboard
kaggle competitions leaderboard -c playground-series-s6e8
```

### 3. **Datasets (if needed)**
```bash
# Download external datasets (if allowed)
kaggle datasets download -d <dataset-slug> -p data/external/
```

---

## Recommended Workflows for This Competition

### Workflow A: Colab GPU Training (Recommended)
```bash
# 1. Create GPU session
colab new -s s6e8-gpu --gpu T4

# 2. Upload data
colab upload -s s6e8-gpu data/train.csv /content/train.csv
colab upload -s s6e8-gpu data/test.csv /content/test.csv

# 3. Install deps
colab install -s s6e8-gpu -r requirements.txt

# 4. Run training
colab exec -s s6e8-gpu -f src/train.py

# 5. Download submission
colab download -s s6e8-gpu /content/submissions/submission.csv submissions/

# 6. Submit to Kaggle
kaggle competitions submit -c playground-series-s6e8 -f submissions/submission.csv -m "Colab T4 run"

# 7. CLEANUP
colab stop -s s6e8-gpu
```

### Workflow B: Kaggle Kernel (Free GPU, persistent)
```bash
# 1. Create kernel directory with kernel-metadata.json
mkdir -p notebooks/kaggle_kernel
cp src/train.py notebooks/kaggle_kernel/
cp data/train.csv data/test.csv notebooks/kaggle_kernel/

# 2. Push and run
kaggle kernels push -p notebooks/kaggle_kernel

# 3. Monitor
kaggle kernels status <your-username>/kaggle_kernel
kaggle kernels logs <your-username>/kaggle_kernel

# 4. Get results
kaggle kernels output <your-username>/kaggle_kernel -p submissions/

# 5. Submit
kaggle competitions submit -c playground-series-s6e8 -f submissions/submission.csv -m "Kaggle kernel run"
```

### Workflow C: Colab One-Shot (No state, auto-cleanup)
```bash
# Single command: provisions, runs, tears down
colab run --gpu T4 src/train.py

# With keep flag to inspect after
colab run --gpu T4 --keep src/train.py
```

---

## Data Flow Diagram
```
Local (this folder)
├── src/train.py, src/predict.py
├── data/train.csv, test.csv
├── config/config.yaml
└── requirements.txt
         │
         ▼ (colab upload / kaggle push)
Remote VM (Colab/Kaggle)
├── /content/train.csv, test.csv
├── /content/src/train.py
├── Run training
├── Save models to /content/models/
├── Save submission to /content/submission.csv
         │
         ▼ (colab download / kaggle kernels output)
Local
├── models/
├── submissions/
         │
         ▼ (kaggle competitions submit)
Kaggle Leaderboard
```

---

## Auth Setup (Run Once)
```bash
# Colab CLI - ADC (recommended for headless)
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory

# Kaggle CLI - already configured if kaggle works
kaggle config view
```

---

## Cost/Quota Notes
| Resource | Free Tier | Notes |
|----------|-----------|-------|
| Colab CPU | Unlimited | Slow for 691K rows |
| Colab T4 | Limited | ~15-30h/month depending on tier |
| Colab A100/H100 | Premium only | Usually fails for free accounts |
| Kaggle GPU | 30h/week | P100/T4, persistent kernels |
| Kaggle TPU | Limited | Need to request access |

---

## Quick Reference Card
```bash
# === DAILY WORKFLOW ===
# 1. Write code locally (this folder)
# 2. Push to remote
colab new -s s6e8 --gpu T4          # or kaggle kernels push
colab upload -s s6e8 data/train.csv /content/train.csv
colab upload -s s6e8 data/test.csv /content/test.csv
colab install -s s6e8 -r requirements.txt
colab exec -s s6e8 -f src/train.py

# 3. Get results
colab download -s s6e8 /content/submissions/*.csv submissions/

# 4. Submit
kaggle competitions submit -c playground-series-s6e8 -f submissions/*.csv -m "msg"

# 5. CLEANUP
colab stop -s s6e8
```