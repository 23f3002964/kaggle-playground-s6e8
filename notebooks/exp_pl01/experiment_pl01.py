"""
PL-01: Iterative Pseudo-labeling (3 rounds)
GPU-accelerated on Colab
"""
# Install dependencies
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'catboost', 'lightgbm', 'xgboost', 'scikit-learn', 'pandas', 'numpy'])

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

# Colab paths
import os
import sys
# Install dependencies first
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'catboost', 'lightgbm', 'xgboost', 'scikit-learn', 'pandas', 'numpy'])

def get_data_dir():
    """Get data directory at runtime"""
    print(f"Checking paths... cwd={os.getcwd()}")
    print(f"/content/train.csv exists: {os.path.exists('/content/train.csv')}")
    print(f"/content/test.csv exists: {os.path.exists('/content/test.csv')}")
    if os.path.exists('/content/train.csv'):
        return Path('/content')
    elif os.path.exists('/content/drive/MyDrive/kaggle/playground-s6e8/train.csv'):
        return Path('/content/drive/MyDrive/kaggle/playground-s6e8')
    elif os.path.exists('/kaggle/input/competitions/playground-series-s6e8/train.csv'):
        return Path('/kaggle/input/competitions/playground-series-s6e8')
    else:
        return Path('../data')

SEED = 42
N_FOLDS = 5
N_ROUNDS = 3
HIGH_THRESH = 0.97
LOW_THRESH = 0.03

def load_data():
    data_dir = get_data_dir()
    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')
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

def add_pool141_features(df):
    df = df.copy()
    df['total_screen_time'] = df['daily_screen_time_hours'] + df['weekend_screen_time']
    df['screen_per_notification'] = df['daily_screen_time_hours'] / (df['notifications_per_day'] + 1e-6)
    df['screen_per_app_open'] = df['daily_screen_time_hours'] / (df['app_opens_per_day'] + 1e-6)
    df['notif_per_app_open'] = df['notifications_per_day'] / (df['app_opens_per_day'] + 1e-6)
    df['leisure_hours'] = df['social_media_hours'] + df['gaming_hours']
    df['leisure_productive_ratio'] = df['leisure_hours'] / (df['work_study_hours'] + 1e-6)
    df['sleep_debt'] = np.maximum(8 - df['sleep_hours'], 0)
    df['screen_per_sleep_hour'] = df['daily_screen_time_hours'] / (df['sleep_hours'] + 1e-6)
    df['age_screen_interaction'] = df['age'] * df['daily_screen_time_hours']
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 50, 100], labels=[0,1,2,3,4]).astype(float)
    df['weekend_weekday_ratio'] = df['weekend_screen_time'] / (df['daily_screen_time_hours'] + 1e-6)
    df['weekend_excess'] = df['weekend_screen_time'] - df['daily_screen_time_hours']
    stress_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Unknown': 1}
    impact_map = {'No': 0, 'Yes': 1, 'Unknown': 0}
    for col, mapping in [('stress_level', stress_map), ('academic_work_impact', impact_map)]:
        if df[col].dtype == 'object':
            df[f'{col}_num'] = df[col].map(mapping)
        else:
            df[f'{col}_num'] = df[col]
    df['stress_screen'] = df['stress_level_num'] * df['daily_screen_time_hours']
    df['impact_screen'] = df['academic_work_impact_num'] * df['daily_screen_time_hours']
    return df

def get_features(train, test):
    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['id', 'addicted_label']]
    train = add_missingness_indicators(train, num_cols)
    test = add_missingness_indicators(test, num_cols)
    train = add_pool141_features(train)
    test = add_pool141_features(test)
    train = preprocess_basic(train)
    test = preprocess_basic(test)
    feature_cols = [c for c in train.columns if c not in ['id', 'addicted_label']]
    X = train[feature_cols]
    y = train['addicted_label']
    X_test = test[feature_cols]
    return X, y, X_test, feature_cols

def train_xgb_gpu(X, y, X_test, params=None):
    """Train XGBoost on GPU"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    test_preds = np.zeros(len(X_test))
    oof = np.zeros(len(X))
    
    params_xgb = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'gpu_hist',
        'gpu_id': 0, 'predictor': 'gpu_predictor',
        'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'min_child_weight': 1,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': SEED,
        'n_jobs': -1, 'verbosity': 0,
    }
    p = {**params_xgb, **(params or {})}
    
    for fold, (tr, val) in enumerate(StratifiedKFold(N_FOLDS, True, SEED).split(X, y)):
        dtrain = xgb.DMatrix(X.iloc[tr], label=y.iloc[tr])
        dval = xgb.DMatrix(X.iloc[val], label=y.iloc[val])
        dtest = xgb.DMatrix(X_test)
        m = xgb.train(p, dtrain, 2000, [(dtrain, 'tr'), (dval, 'val')],
                     early_stopping_rounds=100, verbose_eval=0)
        oof[val] = m.predict(dval, iteration_range=(0, m.best_iteration+1))
        test_preds = m.predict(dtest, iteration_range=(0, m.best_iteration+1))
    
    return oof, test_preds

def train_lgb_gpu(X, y, X_test, params=None):
    """Train LightGBM on GPU"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    test_preds = np.zeros(len(X_test))
    oof = np.zeros(len(X))
    
    params_lgb = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1,
        'seed': SEED, 'n_jobs': -1, 'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0,
    }
    p = {**params_lgb, **(params or {})}
    
    for fold, (tr, val) in enumerate(StratifiedKFold(N_FOLDS, True, SEED).split(X, y)):
        tr_data = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
        val_data = lgb.Dataset(X.iloc[val], label=y.iloc[val], reference=tr_data)
        callbacks = [lgb.early_stopping(100), lgb.log_evaluation(0)]
        m = lgb.train(p, tr_data, 2000, [tr_data, val_data], ['tr', 'val'], callbacks=callbacks)
        oof[val] = m.predict(X.iloc[val], num_iteration=m.best_iteration)
    
    return oof

def pseudo_labeling_round(X, y, X_test, round_num, high_thresh=0.97, low_thresh=0.03):
    """One round of pseudo-labeling"""
    print(f"\n=== Pseudo-labeling Round {round_num} ===")
    
    # Train ensemble on current data
    print("Training ensemble for pseudo-labeling...")
    xgb_oof, xgb_test = train_xgb_gpu(X, y, X_test)
    lgb_oof, lgb_test = train_lgb_gpu(X, y, X_test)
    
    # Average ensemble
    ensemble_test = (xgb_test + lgb_test) / 2
    ensemble_oof = (xgb_oof + lgb_oof) / 2
    
    # Find high-confidence predictions
    high_conf = (ensemble_test >= HIGH_THRESH) | (ensemble_test <= LOW_THRESH)
    n_pseudo = high_conf.sum()
    
    print(f"High-confidence pseudo-labels: {n_pseudo} ({HIGH_THRESH}/{LOW_THRESH})")
    
    if n_pseudo == 0:
        print("No high-confidence samples found")
        return X, y, ensemble_test, ensemble_oof
    
    # Create pseudo-labels
    pseudo_labels = (ensemble_test[high_conf] >= 0.5).astype(int)
    pseudo_X = X_test[high_conf].copy()
    pseudo_y = pd.Series(pseudo_labels, index=pseudo_X.index)
    
    # Combine with original
    X_aug = pd.concat([X, pseudo_X], axis=0)
    y_aug = pd.concat([y, pseudo_y], axis=0)
    
    print(f"Augmented training size: {len(X_aug)} (original: {len(X)} + pseudo: {n_pseudo})")
    
    # Retrain on augmented data
    print("Retraining on augmented data...")
    xgb_oof_aug, xgb_test_aug = train_xgb_gpu(X_aug, y_aug, X_test)
    lgb_oof_aug, lgb_test_aug = train_lgb_gpu(X_aug, y_aug, X_test)
    
    # Evaluate on original data
    ensemble_oof_aug = (xgb_oof_aug[:len(y)] + lgb_oof_aug[:len(y)]) / 2
    auc = roc_auc_score(y, ensemble_oof_aug)
    print(f"Round {round_num} Original data AUC: {auc:.5f}")
    
    return X_aug, y_aug, X_test, (xgb_oof_aug, lgb_oof_aug), (xgb_test, lgb_test)

def main():
    print("="*60)
    print("PL-01: Iterative Pseudo-labeling (3 rounds) - GPU")
    print("="*60)
    
    train, test = load_data()
    print("Preparing features...")
    X, y, X_test, feature_cols = get_features(train, test)
    print(f"Features: {X.shape[1]}, Train: {len(X)}, Test: {len(X_test)}")
    
    # Initial training
    print("\n=== Initial Training ===")
    xgb_oof, xgb_test = train_xgb_gpu(X, y, X_test)
    lgb_oof, lgb_test = train_lgb_gpu(X, y, X_test)
    
    ensemble_oof = (xgb_oof + lgb_oof) / 2
    ensemble_test = (xgb_test + lgb_test) / 2
    
    base_auc = roc_auc_score(y, (xgb_oof + lgb_oof) / 2)
    print(f"Base ensemble AUC: {base_auc:.5f}")
    
    # Iterative pseudo-labeling
    X_curr, y_curr = X.copy(), y.copy()
    test_preds = None
    
    for round_num in range(1, N_ROUNDS + 1):
        X_curr, y_curr, X_test, (xgb_oof, lgb_oof), (xgb_test, lgb_test) = \
            pseudo_labeling_round(X, y, X_test, round_num)
        
        # Keep track of final test predictions
        if round_num == N_ROUNDS:
            xgb_final = xgb_test
            lgb_final = lgb_test
    
    # Final ensemble
    final_test = (xgb_final + lgb_final) / 2
    final_test = np.clip(final_test, 0, 1)
    
    # Generate submission
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    submission = pd.DataFrame({
        'id': sample_sub['id'],
        'addicted_label': final_test
    })
    
    output_dir = Path('/content/drive/MyDrive/kaggle/playground-s6e8/submissions') if Path('/content/drive/MyDrive/kaggle/playground-s6e8').exists() else Path('../submissions')
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / f'submission_pl3rounds_gpu.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to {submission_path}")
    print(f"Shape: {submission.shape}, Mean: {final_test.mean():.5f}, Min: {final_test.min():.5f}, Max: {final_test.max():.5f}")

if __name__ == '__main__':
    main()