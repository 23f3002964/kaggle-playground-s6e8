# Experiment Plan: Closing 0.00019 AUC Gap to Rank #1

## Current State
- Best Public Score: 0.97105 (pool141)
- Target: 0.97124 (Rank #1)
- Gap: 0.00019 AUC

---

## Experiment Matrix

Each experiment runs independently on remote (Kaggle/Colab). Results logged to `logs/experiments.csv`.

| Exp ID | Category | Description | Expected Gain | Priority | Status |
|--------|----------|-------------|---------------|----------|--------|
| **FE-01** | Feature Eng | Interaction features: screen_time × stress, gaming × sleep, notifications/app_opens ratio | +0.00005-0.00010 | High | Pending |
| **FE-02** | Feature Eng | Target encoding for gender/stress/impact (with CV leakage prevention) | +0.00003-0.00008 | High | Pending |
| **FE-03** | Feature Eng | Missingness indicators (is_missing_*) for all numerical cols | +0.00002-0.00005 | High | Pending |
| **FE-04** | Feature Eng | K-means clustering (5-10 clusters) on numerical features → cluster_id feature | +0.00003-0.00007 | Medium | Pending |
| **FE-05** | Feature Eng | Polynomial features (degree=2) for top 5 correlated numerical features | +0.00002-0.00006 | Medium | Pending |
| **FE-06** | Feature Eng | Aggregation features: total_screen = daily + weekend, screen_per_notification, etc. | +0.00003-0.00008 | High | Pending |

| **MD-01** | Model Div | XGBoost baseline (tuned) | +0.00005-0.00012 | High | Pending |
| **MD-02** | Model Div | CatBoost baseline (native categoricals) | +0.00005-0.00012 | High | Pending |
| **MD-03** | Model Div | HistGradientBoosting (sklearn) | +0.00003-0.00008 | Medium | Pending |
| **MD-04** | Model Div | Neural Network (MLP, 2-3 layers) | +0.00003-0.00010 | Medium | Pending |
| **MD-05** | Model Div | Random Forest / ExtraTrees | +0.00002-0.00006 | Low | Pending |

| **ST-01** | Stacking | Level 1: Average of LGB+XGB+CB OOF predictions | +0.00005-0.00010 | High | Pending |
| **ST-02** | Stacking | Level 1: Ridge regression on OOF predictions | +0.00008-0.00015 | High | Pending |
| **ST-03** | Stacking | Level 1: LightGBM meta-learner on OOF predictions | +0.00010-0.00020 | High | Pending |
| **ST-04** | Stacking | Level 2: Stack of stacks (meta-meta) | +0.00005-0.00010 | Medium | Pending |

| **CV-01** | CV Strategy | GroupKFold by age bins (prevent age leakage) | +0.00002-0.00005 | Medium | Pending |
| **CV-02** | CV Strategy | Adversarial validation (train vs test similarity) → drop leaky features | +0.00003-0.00008 | Medium | Pending |
| **CV-03** | CV Strategy | Pseudo-labeling: add high-confidence test preds (prob>0.95 or <0.05) to train | +0.00005-0.00015 | High | Pending |

| **HO-01** | Hyperopt | Optuna: joint optimization of LGB+XGB+CB (shared feature params) | +0.00005-0.00012 | High | Pending |
| **HO-02** | Hyperopt | Optuna: optimize stacking meta-learner hyperparams | +0.00003-0.00008 | Medium | Pending |

| **EN-01** | Ensemble | Greedy ensemble selection (forward/backward) on public LB | +0.00003-0.00008 | High | Pending |
| **EN-02** | Ensemble | Correlation-based diversity selection (keep uncorrelated models) | +0.00002-0.00006 | Medium | Pending |

---

## Execution Order (Dependency Graph)

```
Phase 1: Feature Engineering (run in parallel)
  FE-01 → FE-06 (all feed into model training)

Phase 2: Model Diversity (run in parallel, use best FE set)
  MD-01 → MD-05

Phase 3: Stacking (depends on Phase 2 OOF preds)
  ST-01 → ST-03 (ST-04 optional)

Phase 4: CV Strategy (can run anytime)
  CV-01 → CV-03

Phase 5: Hyperopt (after best model family identified)
  HO-01 → HO-02

Phase 6: Final Ensemble (depends on all above)
  EN-01 → EN-02
```

---

## Remote Execution Protocol

**Before EVERY GPU experiment:**
1. I will ask: **"Kaggle kernel or Colab for this run?"**
2. You choose → I generate the exact commands
3. You execute → Share results → I log and decide next

---

## Logging Format (`logs/experiments.csv`)

```csv
exp_id,category,description,cv_auc,public_lb,private_lb,compute,time_min,notes,status
FE-01,Feature Eng,"screen_time × stress interaction",0.97012,,Colab T4,15,baseline+interactions,completed
```

---

## Success Criteria

| Milestone | Public AUC | Rank Estimate |
|-----------|------------|---------------|
| Baseline | 0.97105 | #33 |
| +FE | ≥0.97115 | #15-20 |
| +Model Div | ≥0.97120 | #5-10 |
| +Stacking | ≥0.97124 | #1-3 |
| **Target** | **≥0.97124** | **#1** |

---

## Quick Start Commands (Copy-Paste)

### For Colab:
```bash
colab new -s s6e8-expXX --gpu T4
colab upload -s s6e8-expXX data/train.csv /content/train.csv
colab upload -s s6e8-expXX data/test.csv /content/test.csv
colab upload -s s6e8-expXX src/experiment_XX.py /content/experiment_XX.py
colab install -s s6e8-expXX -r requirements.txt
colab exec -s s6e8-expXX -f /content/experiment_XX.py
colab download -s s6e8-expXX /content/experiment_XX_results.csv logs/
colab stop -s s6e8-expXX
```

### For Kaggle Kernel:
```bash
mkdir -p notebooks/exp_XX
cp src/experiment_XX.py notebooks/exp_XX/
cp data/train.csv data/test.csv notebooks/exp_XX/
kaggle kernels init -p notebooks/exp_XX
kaggle kernels push -p notebooks/exp_XX
kaggle kernels output <slug> -p logs/
```

---

## Next Action

**Run FE-01 to FE-06 in parallel** (feature engineering experiments). Each tests one feature set added to baseline LightGBM.

**Question for you:** Kaggle kernel or Colab for the feature engineering batch?