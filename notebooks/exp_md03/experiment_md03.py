"""
Model Diversity Experiment: MD-03
HistGradientBoosting (sklearn) with FE-03 features
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

# Kaggle kernel: competition data is at /kaggle/input/competitions/playground-series-s6e8/
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
    df = df.copy()
    for col in num_cols:
        df[f'{col}_was_missing'] = df[col].isna().astype(int)
    return df

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
        print(f"  Fold {fold+1}: AUC = {fold_auc:.5f}")
    
    cv_auc = roc_auc_score(y, oof_preds)
    print(f"  CV AUC: {cv_auc:.5f} (+/- {np.std(fold_scores):.5f})")
    return cv_auc, oof_preds

def main():
    print("="*60)
    print("MD-03: HistGradientBoosting (sklearn) (FE-03 Features)")
    print("="*60)
    
    train, test = load_data()
    
    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['id', 'addicted_label']]
    
    print("Adding FE-03 missingness indicators...")
    train = add_missingness_indicators(train, num_cols)
    test = add_missingness_indicators(test, num_cols)
    
    train = preprocess_basic(train)
    test = preprocess_basic(test)
    
    feature_cols = [c for c in train.columns if c not in ['id', 'addicted_label']]
    X = train[feature_cols]
    y = train['addicted_label']
    
    missing_indicator_cols = [c for c in feature_cols if c.endswith('_was_missing')]
    print(f"Features: {len(feature_cols)} (added {len(missing_indicator_cols)} missingness indicators)")
    print(f"Train samples: {len(X)}")
    
    print("\nTraining HistGradientBoosting with 5-fold CV...")
    cv_auc, oof_preds = train_hgb_cv(X, y)
    
    results = pd.DataFrame({
        'exp_id': ['MD-03'],
        'category': ['Model Div'],
        'description': ['HistGradientBoosting (sklearn) with FE-03 missingness indicators'],
        'cv_auc': [cv_auc],
        'n_features': [len(feature_cols)],
        'status': ['completed']
    })
    
    logs_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('../logs')
    logs_dir.mkdir(exist_ok=True)
    exp_file = logs_dir / 'experiments.csv'
    if exp_file.exists():
        existing = pd.read_csv(exp_file)
        results = pd.concat([existing, results], ignore_index=True)
    results.to_csv(exp_file, index=False)
    print(f"\nResults saved to {exp_file}")
    
    oof_df = pd.DataFrame({
        'id': train['id'],
        'md03_oof': oof_preds
    })
    oof_path = Path('/kaggle/working/md03_oof.csv') if Path('/kaggle/working').exists() else Path('../models/md03_oof.csv')
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to {oof_path}")

if __name__ == '__main__':
    main()