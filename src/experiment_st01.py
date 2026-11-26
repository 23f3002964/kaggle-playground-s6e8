"""
Stacking Experiment: ST-01
Simple average of best model OOF predictions - recomputed internally
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import HistGradientBoostingClassifier

# Kaggle kernel paths
import os
if os.path.exists('/kaggle/input/competitions/playground-series-s6e8/train.csv'):
    DATA_DIR = Path('/kaggle/input/competitions/playground-series-s6e8')
elif os.path.exists('/kaggle/working/train.csv'):
    DATA_DIR = Path('/kaggle/working')
else:
    DATA_DIR = Path('../data')

SEED = 42
N_FOLDS = 5

def load_data():
    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    return train, test

def preprocess_basic(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['id', 'addicted_label']]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    cat_cols = ['gender', 'stress_level', 'academic_work_impact']
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def add_missingness_indicators(df, num_cols):
    """FE-03: Missingness indicators"""
    df = df.copy()
    for col in num_cols:
        df[f'{col}_was_missing'] = df[col].isna().astype(int)
    return df

def get_features(train, test):
    """Prepare features with FE-03 missingness indicators"""
    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['id', 'addicted_label']]
    
    train = add_missingness_indicators(train, num_cols)
    test = add_missingness_indicators(test, num_cols)
    
    train = preprocess_basic(train)
    test = preprocess_basic(test)
    
    feature_cols = [c for c in train.columns if c not in ['id', 'addicted_label']]
    X = train[feature_cols]
    y = train['addicted_label']
    X_test = test[feature_cols]
    
    return X, y, X_test

def train_lgb_cv(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    fold_scores = []
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': SEED,
        'n_jobs': -1,
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_scores.append(fold_auc)
        print(f"  LGB Fold {fold+1}: AUC = {fold_auc:.5f}")
    
    cv_auc = roc_auc_score(y, oof_preds)
    print(f"  LGB CV AUC: {cv_auc:.5f} (+/- {np.std(fold_scores):.5f})")
    return cv_auc, oof_preds

def train_xgb_cv(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    fold_scores = []
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': SEED,
        'n_jobs': -1,
        'verbosity': 0,
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dtrain, 'train'), (dval, 'valid')],
            early_stopping_rounds=100,
            verbose_eval=False
        )
        
        oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_scores.append(fold_auc)
        print(f"  XGB Fold {fold+1}: AUC = {fold_auc:.5f}")
    
    cv_auc = roc_auc_score(y, oof_preds)
    print(f"  XGB CV AUC: {cv_auc:.5f} (+/- {np.std(fold_scores):.5f})")
    return cv_auc, oof_preds

def train_hgb_cv(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    fold_scores = []
    
    params = {
        'loss': 'log_loss',
        'learning_rate': 0.05,
        'max_iter': 2000,
        'max_depth': 6,
        'min_samples_leaf': 20,
        'l2_regularization': 1.0,
        'max_bins': 255,
        'random_state': SEED,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 100,
        'tol': 1e-4,
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = HistGradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_scores.append(fold_auc)
        print(f"  HGB Fold {fold+1}: AUC = {fold_auc:.5f}")
    
    cv_auc = roc_auc_score(y, oof_preds)
    print(f"  HGB CV AUC: {cv_auc:.5f} (+/- {np.std(fold_scores):.5f})")
    return cv_auc, oof_preds

def train_cb_cv(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    fold_scores = []
    
    cat_features = [X.columns.get_loc(c) for c in ['gender', 'stress_level', 'academic_work_impact'] if c in X.columns]
    
    params = {
        'objective': 'Logloss',
        'eval_metric': 'AUC',
        'iterations': 2000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'subsample': 0.8,
        'colsample_bylevel': 0.8,
        'random_seed': SEED,
        'verbose': False,
        'thread_count': -1,
        'early_stopping_rounds': 100,
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_scores.append(fold_auc)
        print(f"  CB Fold {fold+1}: AUC = {fold_auc:.5f}")
    
    cv_auc = roc_auc_score(y, oof_preds)
    print(f"  CB CV AUC: {cv_auc:.5f} (+/- {np.std(fold_scores):.5f})")
    return cv_auc, oof_preds

def evaluate_ensembles(oof_dict, y_true):
    """Evaluate different ensemble combinations"""
    model_names = list(oof_dict.keys())
    common_idx = None
    for name, preds in oof_dict.items():
        if common_idx is None:
            common_idx = preds.index
        else:
            common_idx = common_idx.intersection(preds.index)
    
    if common_idx is None:
        raise ValueError("No predictions to ensemble")
    
    print(f"Common index size: {len(common_idx)}")
    
    y = y_true.loc[common_idx]
    aligned_preds = {name: preds.loc[common_idx] for name, preds in oof_dict.items()}
    
    results = []
    
    # Individual models
    for name, preds in aligned_preds.items():
        auc = roc_auc_score(y, preds)
        results.append(('single', name, auc))
        print(f"  {name}: {auc:.5f}")
    
    # Simple average combinations
    # Top 2: XGBoost + LightGBM
    top2 = ['xgb', 'lgb']
    if all(m in aligned_preds for m in top2):
        avg2 = np.mean([aligned_preds[m] for m in top2], axis=0)
        auc2 = roc_auc_score(y, avg2)
        results.append(('avg_top2', '+'.join(top2), auc2))
        print(f"  avg_top2: {auc2:.5f}")
    
    # Top 3: XGBoost + LightGBM + HGB
    top3 = ['xgb', 'lgb', 'hgb']
    if all(m in aligned_preds for m in top3):
        avg3 = np.mean([aligned_preds[m] for m in top3], axis=0)
        auc3 = roc_auc_score(y, avg3)
        results.append(('avg_top3', '+'.join(top3), auc3))
        print(f"  avg_top3: {auc3:.5f}")
    
    # All 4
    avg4 = np.mean([aligned_preds[m] for m in model_names], axis=0)
    auc4 = roc_auc_score(y, avg4)
    results.append(('avg_all4', '+'.join(model_names), auc4))
    print(f"  avg_all4: {auc4:.5f}")
    
    # Weighted average
    weights = {'xgb': 0.96384, 'lgb': 0.96323, 'hgb': 0.96315, 'cb': 0.96112}
    total_w = sum(weights[m] for m in model_names)
    norm_weights = {m: weights[m]/total_w for m in model_names}
    weighted = sum(norm_weights[m] * aligned_preds[m] for m in model_names)
    auc_w = roc_auc_score(y, weighted)
    results.append(('weighted', '+'.join(model_names), auc_w))
    print(f"  weighted: {auc_w:.5f}")
    
    return results, y, aligned_preds

def main():
    print("="*60)
    print("ST-01: Simple Average Stacking (Internal OOF Computation)")
    print("="*60)
    
    train, test = load_data()
    
    print("Preparing features with FE-03 missingness indicators...")
    X, y, X_test = get_features(train, test)
    print(f"Features: {X.shape[1]}, Train samples: {len(X)}")
    
    # Train top 4 models and get OOF predictions
    print("\nTraining LightGBM...")
    lgb_auc, lgb_oof = train_lgb_cv(X, y)
    
    print("\nTraining XGBoost...")
    xgb_auc, xgb_oof = train_xgb_cv(X, y)
    
    print("\nTraining HistGradientBoosting...")
    hgb_auc, hgb_oof = train_hgb_cv(X, y)
    
    print("\nTraining CatBoost...")
    cb_auc, cb_oof = train_cb_cv(X, y)
    
    # Collect OOF predictions
    oof_dict = {
        'lgb': pd.Series(lgb_oof, index=train['id']),
        'xgb': pd.Series(xgb_oof, index=train['id']),
        'hgb': pd.Series(hgb_oof, index=train['id']),
        'cb': pd.Series(cb_oof, index=train['id']),
    }
    
    print("\nEvaluating ensembles...")
    y_true = train.set_index('id')['addicted_label']
    results, y, aligned_preds = evaluate_ensembles(oof_dict, y_true)
    
    # Save results
    results_df = pd.DataFrame(results, columns=['type', 'models', 'cv_auc'])
    results_df['exp_id'] = 'ST-01'
    results_df['category'] = 'Stacking'
    results_df['description'] = 'Simple average stacking of top 4 models (recomputed internally)'
    results_df['status'] = 'completed'
    
    logs_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('../logs')
    logs_dir.mkdir(exist_ok=True)
    exp_file = logs_dir / 'experiments.csv'
    if exp_file.exists():
        existing = pd.read_csv(exp_file)
        results_df = pd.concat([existing, results_df], ignore_index=True)
    results_df.to_csv(exp_file, index=False)
    print(f"\nResults saved to {exp_file}")
    
    # Save ensemble OOF for meta-learning
    ensemble_df = pd.DataFrame({'id': y.index})
    for name, preds in aligned_preds.items():
        ensemble_df[name] = preds.values
    ensemble_df['avg_top3'] = np.mean([aligned_preds['xgb'], aligned_preds['lgb'], aligned_preds['hgb']], axis=0)
    
    ensemble_path = logs_dir / 'st01_ensemble_oof.csv'
    ensemble_df.to_csv(ensemble_path, index=False)
    print(f"Ensemble OOF saved to {ensemble_path}")

if __name__ == '__main__':
    main()