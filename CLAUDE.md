# Project Rules: Playground Series S6E8 - Smartphone Addiction
# **SCOPE: THIS FOLDER ONLY** — Does not apply to other projects

## Competition Rules (Kaggle)
- **No external data** — Only provided train/test data
- **Submission limit**: 5/day
- **Metric**: AUC (ROC-AUC)
- **Deadline**: 2026-08-31 23:59:00 UTC
- **Code must be reproducible** — Save seeds, configs, env

## Development Rules

### Code Style
- **Ponytail mode**: Minimal, stdlib-first, no over-engineering
- **One script per concern**: train.py, predict.py, preprocess.py
- **Config in YAML** — No hardcoded params in code
- **Type hints** on all functions

### Data
- **Never modify raw data** in `data/` — Process in memory/scripts
- **Missing values**: Median (num), "Unknown" (cat) — Document in config
- **Categorical encoding**: LabelEncoder for tree models
- **No leakage** — Fit encoders/scalers only on train, apply to test

### Modeling
- **CV is mandatory** — 5-fold StratifiedKFold, seed=42
- **OOF predictions saved** for stacking/analysis
- **Models saved** to `models/` with fold number
- **Submissions to** `submissions/` with CV score in filename

### Experiment Tracking
- **Log every run**: params, CV score, OOF AUC, time
- **Use MLflow** or simple CSV log in `logs/experiments.csv`
- **Tag best model** per experiment

### Git
- **Commit before major changes** — `git add -A && git commit -m "msg"`
- **Large files** (data, models) in `.gitignore`
- **Submissions not committed** unless it's the final one

### Quality Gates (before claiming done)
```bash
# 1. Train runs without error
python src/train.py

# 2. Predict generates valid submission
python src/predict.py

# 3. Submission format correct
head -3 submissions/*.csv

# 4. CV score logged
cat logs/experiments.csv | tail -1
```

## Quick Commands
```bash
# Train baseline
python src/train.py

# Generate submission from saved models
python src/predict.py

# View experiments
cat logs/experiments.csv
```

## Execution Rules (CRITICAL)

### 1. NO LOCAL EXECUTION
- **Nothing runs on this laptop** — Zero local training/inference
- **NO HEAVY TASKS AT ALL** (blending, CV sweeps, stacking, Optuna, feature scans) — RAM sirf 13GB hai, OOM hota hai
- **Heavy CPU bhi Kaggle kernel pe hi run karo** — code yahan likho → push → remote result
- **Allowed compute**: Kaggle kernels, Colab via CLI, or MCP servers only
- Code written here → pushed → run remotely

### 2. Todo List Protocol
- **Before ANY work**: Update todo list (status: in_progress)
- **On interruption**: Todo list persists in background, survives command failures
- **On completion**: Mark completed, add follow-ups discovered

### 3. Kaggle Submission Flow
- **Never wait** after `kaggle competitions submit`
- **User controls notebook execution** — I don't poll, don't wait, don't assume
- **User tells me** when notebook runs, when results are ready

### 4. GPU Execution Protocol (MANDATORY)
- **Before ANY GPU work**: I MUST ask → "Kaggle kernel or Colab for this run?"
- **You decide** — I execute the commands for your choice
- **No assumption** — Every GPU experiment requires explicit confirmation

### 5. Compute Selection Rule (AUTO)
- **CPU-only workloads** (feature engineering, CV, blending, Optuna CPU trials) → **Kaggle Kernel** (free CPU, 30h/week GPU quota saved)
- **GPU workloads** (deep learning, large XGB/LGB/CB GPU training, neural nets) → **Colab** (T4/A100 GPU)
- **Default**: If unsure → I ask

### 6. Colab GPU Execution Protocol
- **All GPU work → Colab** (not Kaggle)
- **Kaggle untouched** unless CPU-only or submission
- **Long Colab tasks MUST have progressive progress bars:**
  - Show epoch/step progress + ETA
  - Log parseable for status check in next cell
  - No blocking waits - check status in next cell
  - Example: `tqdm` with `postfix` dict + log parsing cell

## Ponytail Reminders
- Don't add features until baseline works
- Don't tune hyperparams until CV is stable
- One model type first (LightGBM), then ensemble
- Delete dead code, don't comment it