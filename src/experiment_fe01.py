"""
Feature Engineering Experiment: FE-01
Interaction features: screen_time × stress, gaming × sleep, notifications/app_opens ratio
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
    import os
    print(f"Working dir: {os.getcwd()}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"/kaggle exists: {os.path.exists('/kaggle')}")
    print(f"/kaggle/input exists: {os.path.exists('/kaggle/input')}")
    if os.path.exists('/kaggle/input'):
        print(f"/kaggle/input contents: {os.listdir('/kaggle/input')}")
    print(f"Files in working dir: {os.listdir('/kaggle/working') if os.path.exists('/kaggle/working') else 'N/A'}")
    print(f"Files in input dir: {os.listdir('/kaggle/input/playground-series-s6e8') if os.path.exists('/kaggle/input/playground-series-s6e8') else 'N/A'}")
    print(f"DATA_DIR exists: {DATA_DIR.exists()}")
    print(f"Train exists: {(DATA_DIR / 'train.csv').exists()}")
    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    return train, test

def preprocess_basic(df):
    """Basic preprocessing from baseline"""
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

def add_fe01_features(df):
    """FE-01: Interaction features"""
    df = df.copy()
    # screen_time × stress (encoded)
    stress_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Unknown': 1}
    df['stress_encoded'] = df['stress_level'].map(stress_map)
    df['screen_stress_interaction'] = df['daily_screen_time_hours'] * df['stress_encoded']
    
    # gaming × sleep
    df['gaming_sleep_ratio'] = df['gaming_hours'] / (df['sleep_hours'] + 1e-6)
    df['gaming_sleep_interaction'] = df['gaming_hours'] * df['sleep_hours']
    
    # notifications / app_opens ratio
    df['notif_per_app_open'] = df['notifications_per_day'] / (df['app_opens_per_day'] + 1e-6)
    df['app_open_per_notif'] = df['app_opens_per_day'] / (df['notifications_per_day'] + 1e-6)
    
    # total screen time
    df['total_screen_time'] = df['daily_screen_time_hours'] + df['weekend_screen_time']
    
    # screen time per hour awake
    df['awake_hours'] = 24 - df['sleep_hours']
    df['screen_per_awake_hour'] = df['daily_screen_time_hours'] / (df['awake_hours'] + 1e-6)
    
    # work/life balance proxy
    df['work_study_per_sleep'] = df['work_study_hours'] / (df['sleep_hours'] + 1e-6)
    df['leisure_vs_work'] = (df['social_media_hours'] + df['gaming_hours']) / (df['work_study_hours'] + 1e-6)
    
    # age interactions
    df['age_screen_interaction'] = df['age'] * df['daily_screen_time_hours']
    df['age_gaming_interaction'] = df['age'] * df['gaming_hours']
    
    # drop helper column
    df = df.drop(['stress_encoded'], axis=1)
    return df

def train_lgb_cv(X, y):
    """Train LightGBM with 5-fold CV, return OOF preds and CV score"""
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
    print("FE-01: Interaction Features Experiment")
    print("="*60)
    
    train, test = load_data()
    
    # Combine for consistent preprocessing
    train['is_train'] = 1
    test['is_train'] = 0
    test['addicted_label'] = -1
    df = pd.concat([train, test], axis=0, ignore_index=True)
    
    # Basic preprocessing
    df = preprocess_basic(df)
    
    # Add FE-01 features
    print("Adding FE-01 interaction features...")
    df = add_fe01_features(df)
    
    # Split back
    train_proc = df[df['is_train'] == 1].drop(['is_train'], axis=1).reset_index(drop=True)
    test_proc = df[df['is_train'] == 0].drop(['is_train', 'addicted_label'], axis=1).reset_index(drop=True)
    
    feature_cols = [c for c in train_proc.columns if c not in ['id', 'addicted_label']]
    X = train_proc[feature_cols]
    y = train_proc['addicted_label']
    
    print(f"Features: {len(feature_cols)} (added {len(feature_cols) - 13} new)")
    print(f"Train samples: {len(X)}")
    
    # Train with CV
    print("\nTraining LightGBM with 5-fold CV...")
    cv_auc, oof_preds = train_lgb_cv(X, y)
    
    # Save results
    results = pd.DataFrame({
        'exp_id': ['FE-01'],
        'category': ['Feature Eng'],
        'description': ['Interaction features: screen_stress, gaming_sleep, notif_ratios, total_screen, age_interactions'],
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
    
    # Save OOF for potential stacking
    oof_df = pd.DataFrame({
        'id': train['id'],
        'fe01_oof': oof_preds
    })
    # Save to working dir in kernel, fallback to local models/
    oof_path = Path('/kaggle/working/fe01_oof.csv') if Path('/kaggle/working').exists() else Path('../models/fe01_oof.csv')
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to {oof_path}")

if __name__ == '__main__':
    main()