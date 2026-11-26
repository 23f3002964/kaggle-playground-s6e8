"""
Feature Engineering Experiment: FE-05
Polynomial features (degree=2) for top 5 correlated numerical features
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
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
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def get_top_correlated_features(train_df, target_col, top_k=5):
    """Get top k numerical features by correlation with target"""
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['id', target_col]]
    
    corrs = train_df[num_cols + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
    top_features = corrs.index[1:top_k+1].tolist()  # Skip target itself
    return top_features

def add_polynomial_features(train_df, test_df, feature_cols, degree=2):
    """Add polynomial features for selected columns"""
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    
    # Fit on train
    X_train_poly = poly.fit_transform(train_df[feature_cols])
    X_test_poly = poly.transform(test_df[feature_cols])
    
    # Get feature names
    poly_feature_names = poly.get_feature_names_out(feature_cols)
    
    # Add polynomial features (excluding original features which are already present)
    for i, name in enumerate(poly_feature_names):
        if name not in feature_cols:  # Only add new interaction/squared terms
            train_df[f'poly_{name}'] = X_train_poly[:, i]
            test_df[f'poly_{name}'] = X_test_poly[:, i]
    
    return train_df, test_df

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
    print("FE-05: Polynomial Features (degree=2) for Top 5 Correlated Features")
    print("="*60)
    
    train, test = load_data()
    
    # Basic preprocessing first
    train = preprocess_basic(train)
    test = preprocess_basic(test)
    
    # Get top 5 correlated numerical features
    top_features = get_top_correlated_features(train, 'addicted_label', top_k=5)
    print(f"Top 5 correlated features: {top_features}")
    
    # Add polynomial features
    print("Generating polynomial features (degree=2)...")
    train, test = add_polynomial_features(train, test, top_features, degree=2)
    
    feature_cols = [c for c in train.columns if c not in ['id', 'addicted_label']]
    X = train[feature_cols]
    y = train['addicted_label']
    
    poly_cols = [c for c in feature_cols if c.startswith('poly_')]
    print(f"Features: {len(feature_cols)} (added {len(poly_cols)} polynomial features)")
    print(f"Polynomial features: {poly_cols[:10]}...")
    print(f"Train samples: {len(X)}")
    
    print("\nTraining LightGBM with 5-fold CV...")
    cv_auc, oof_preds = train_lgb_cv(X, y)
    
    results = pd.DataFrame({
        'exp_id': ['FE-05'],
        'category': ['Feature Eng'],
        'description': ['Polynomial features (degree=2) for top 5 target-correlated numerical features'],
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
        'fe05_oof': oof_preds
    })
    oof_path = Path('/kaggle/working/fe05_oof.csv') if Path('/kaggle/working').exists() else Path('../models/fe05_oof.csv')
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to {oof_path}")

if __name__ == '__main__':
    main()