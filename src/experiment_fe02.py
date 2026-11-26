"""
Feature Engineering Experiment: FE-02
Target encoding for gender/stress/impact with CV leakage prevention
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

# Kaggle kernel: competition data is at /kaggle/input/competitions/playground-series-s6e8/
# Script runs from /kaggle/src/, working dir is /kaggle/working/
import os
# Check competition data path first (mounted by competition_sources)
if os.path.exists('/kaggle/input/competitions/playground-series-s6e8/train.csv'):
    DATA_DIR = Path('/kaggle/input/competitions/playground-series-s6e8')
# Check working dir (if files uploaded with kernel)
elif os.path.exists('/kaggle/working/train.csv'):
    DATA_DIR = Path('/kaggle/working')
else:
    DATA_DIR = Path('../data')  # fallback for local
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
    return df

def add_target_encoding_cv(train_df, test_df, cat_cols, target_col, n_folds=5, seed=42):
    """Target encoding with CV to prevent leakage"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    global_mean = train_df[target_col].mean()
    
    for col in cat_cols:
        # Train: CV target encoding
        train_df[f'{col}_target_enc'] = 0.0
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[target_col])):
            train_fold = train_df.iloc[train_idx]
            val_fold = train_df.iloc[val_idx]
            
            # Compute mean per category on train fold
            cat_means = train_fold.groupby(col)[target_col].mean()
            
            # Apply to validation fold
            train_df.loc[val_idx, f'{col}_target_enc'] = val_fold[col].map(cat_means).fillna(global_mean)
        
        # Test: use full train mean
        cat_means_full = train_df.groupby(col)[target_col].mean()
        test_df[f'{col}_target_enc'] = test_df[col].map(cat_means_full).fillna(global_mean)
    
    return train_df, test_df

def add_fe02_features(df):
    """FE-02: Target encoded categoricals"""
    return df  # Encoding happens in main()

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
        print(f"  Fold {fold+1}: AUC = {fold_auc:.5f}")
    
    cv_auc = roc_auc_score(y, oof_preds)
    print(f"  CV AUC: {cv_auc:.5f} (+/- {np.std(fold_scores):.5f})")
    return cv_auc, oof_preds

def main():
    print("="*60)
    print("FE-02: Target Encoding for Categoricals (CV-safe)")
    print("="*60)
    
    train, test = load_data()
    
    # Basic preprocessing (keep original categoricals for encoding)
    train = preprocess_basic(train)
    test = preprocess_basic(test)
    
    cat_cols = ['gender', 'stress_level', 'academic_work_impact']
    
    # Target encoding with CV
    print("Applying CV-safe target encoding...")
    train, test = add_target_encoding_cv(train, test, cat_cols, 'addicted_label')
    
    # Label encode original categoricals for tree models
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0)
        le.fit(combined.astype(str))
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
    
    feature_cols = [c for c in train.columns if c not in ['id', 'addicted_label']]
    X = train[feature_cols]
    y = train['addicted_label']
    
    print(f"Features: {len(feature_cols)} (added {len(cat_cols)} target encodings)")
    print(f"Train samples: {len(X)}")
    
    print("\nTraining LightGBM with 5-fold CV...")
    cv_auc, oof_preds = train_lgb_cv(X, y)
    
    results = pd.DataFrame({
        'exp_id': ['FE-02'],
        'category': ['Feature Eng'],
        'description': ['CV-safe target encoding for gender, stress_level, academic_work_impact'],
        'cv_auc': [cv_auc],
        'n_features': [len(feature_cols)],
        'status': ['completed']
    })
    
    logs_dir = Path('../logs')
    logs_dir.mkdir(exist_ok=True)
    exp_file = logs_dir / 'experiments.csv'
    if exp_file.exists():
        existing = pd.read_csv(exp_file)
        results = pd.concat([existing, results], ignore_index=True)
    results.to_csv(exp_file, index=False)
    print(f"\nResults saved to {exp_file}")
    
    oof_df = pd.DataFrame({
        'id': train['id'],
        'fe02_oof': oof_preds
    })
    oof_path = Path('/kaggle/working/fe02_oof.csv') if Path('/kaggle/working').exists() else Path('../models/fe02_oof.csv')
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to {oof_path}")

if __name__ == '__main__':
    main()