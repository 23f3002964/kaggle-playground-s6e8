"""
Feature Engineering Experiment: FE-04
K-means clustering on numerical features -> cluster_id feature
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
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
N_CLUSTERS = 8

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

def add_kmeans_clusters(train_df, test_df, n_clusters=8):
    """Fit KMeans on train numerical features, predict on both"""
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['id', 'addicted_label']]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[num_cols])
    X_test_scaled = scaler.transform(test_df[num_cols])
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(X_train_scaled)
    test_clusters = kmeans.predict(X_test_scaled)
    
    train_df[f'kmeans_cluster_{n_clusters}'] = train_clusters
    test_df[f'kmeans_cluster_{n_clusters}'] = test_clusters
    
    # Also add distance to cluster centers as features
    train_dist = kmeans.transform(X_train_scaled)
    test_dist = kmeans.transform(X_test_scaled)
    
    for i in range(n_clusters):
        train_df[f'kmeans_dist_{i}'] = train_dist[:, i]
        test_df[f'kmeans_dist_{i}'] = test_dist[:, i]
    
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
    print(f"FE-04: K-Means Clustering (n_clusters={N_CLUSTERS})")
    print("="*60)
    
    train, test = load_data()
    
    # Basic preprocessing first
    train = preprocess_basic(train)
    test = preprocess_basic(test)
    
    # Add KMeans clusters
    print(f"Fitting KMeans with {N_CLUSTERS} clusters...")
    train, test = add_kmeans_clusters(train, test, N_CLUSTERS)
    
    feature_cols = [c for c in train.columns if c not in ['id', 'addicted_label']]
    X = train[feature_cols]
    y = train['addicted_label']
    
    cluster_cols = [c for c in feature_cols if c.startswith('kmeans_')]
    print(f"Features: {len(feature_cols)} (added {len(cluster_cols)} cluster features)")
    print(f"Cluster features: {cluster_cols[:5]}...")
    print(f"Train samples: {len(X)}")
    
    print("\nTraining LightGBM with 5-fold CV...")
    cv_auc, oof_preds = train_lgb_cv(X, y)
    
    results = pd.DataFrame({
        'exp_id': ['FE-04'],
        'category': ['Feature Eng'],
        'description': [f'K-means clustering (n={N_CLUSTERS}) on numerical features + distances to centers'],
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
        'fe04_oof': oof_preds
    })
    oof_path = Path('/kaggle/working/fe04_oof.csv') if Path('/kaggle/working').exists() else Path('../models/fe04_oof.csv')
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to {oof_path}")

if __name__ == '__main__':
    main()