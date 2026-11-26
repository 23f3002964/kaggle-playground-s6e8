"""
Feature Engineering Experiment: FE-06
Aggregation features: total_screen, screen_per_notification, etc.
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
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def add_aggregation_features(df):
    """Add aggregated/ratio features"""
    df = df.copy()
    
    # Total screen time
    df['total_screen_time'] = df['daily_screen_time_hours'] + df['weekend_screen_time']
    df['avg_daily_screen'] = (df['daily_screen_time_hours'] * 5 + df['weekend_screen_time'] * 2) / 7
    
    # Screen time ratios
    df['screen_per_notification'] = df['daily_screen_time_hours'] / (df['notifications_per_day'] + 1e-6)
    df['screen_per_app_open'] = df['daily_screen_time_hours'] / (df['app_opens_per_day'] + 1e-6)
    df['notif_per_app_open'] = df['notifications_per_day'] / (df['app_opens_per_day'] + 1e-6)
    
    # Leisure vs productive time
    df['leisure_hours'] = df['social_media_hours'] + df['gaming_hours']
    df['productive_hours'] = df['work_study_hours']
    df['leisure_productive_ratio'] = df['leisure_hours'] / (df['productive_hours'] + 1e-6)
    df['total_wake_hours'] = df['leisure_hours'] + df['productive_hours'] + df['sleep_hours']
    
    # Sleep features
    df['sleep_debt'] = np.maximum(8 - df['sleep_hours'], 0)
    df['screen_per_sleep_hour'] = df['daily_screen_time_hours'] / (df['sleep_hours'] + 1e-6)
    
    # Age-based features
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 50, 100], labels=[0, 1, 2, 3, 4]).astype(float)
    df['age_screen_interaction'] = df['age'] * df['daily_screen_time_hours']
    df['age_leisure_interaction'] = df['age'] * df['leisure_hours']
    
    # Intensity features
    df['digital_intensity'] = (df['notifications_per_day'] + df['app_opens_per_day']) / (df['daily_screen_time_hours'] + 1e-6)
    df['notification_intensity'] = df['notifications_per_day'] / (df['daily_screen_time_hours'] + 1e-6)
    
    # Weekend vs weekday pattern
    df['weekend_weekday_ratio'] = df['weekend_screen_time'] / (df['daily_screen_time_hours'] + 1e-6)
    df['weekend_excess'] = df['weekend_screen_time'] - df['daily_screen_time_hours']
    
    # Stress-level numeric mapping
    stress_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Unknown': 1}
    if df['stress_level'].dtype == 'object':
        df['stress_numeric'] = df['stress_level'].map(stress_map)
    else:
        df['stress_numeric'] = df['stress_level']
    
    df['stress_screen_interaction'] = df['stress_numeric'] * df['daily_screen_time_hours']
    df['stress_leisure_interaction'] = df['stress_numeric'] * df['leisure_hours']
    
    # Academic impact numeric
    impact_map = {'No': 0, 'Yes': 1, 'Unknown': 0}
    if df['academic_work_impact'].dtype == 'object':
        df['impact_numeric'] = df['academic_work_impact'].map(impact_map)
    else:
        df['impact_numeric'] = df['academic_work_impact']
    
    df['impact_screen_interaction'] = df['impact_numeric'] * df['daily_screen_time_hours']
    df['impact_leisure_interaction'] = df['impact_numeric'] * df['leisure_hours']
    
    # Drop helper columns
    helper_cols = ['stress_numeric', 'impact_numeric']
    df = df.drop([c for c in helper_cols if c in df.columns], axis=1)
    
    return df

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
    print("FE-06: Aggregation & Ratio Features")
    print("="*60)
    
    train, test = load_data()
    
    # Basic preprocessing first
    train = preprocess_basic(train)
    test = preprocess_basic(test)
    
    # Add aggregation features
    print("Adding aggregation and ratio features...")
    train = add_aggregation_features(train)
    test = add_aggregation_features(test)
    
    feature_cols = [c for c in train.columns if c not in ['id', 'addicted_label']]
    X = train[feature_cols]
    y = train['addicted_label']
    
    agg_cols = [c for c in feature_cols if c not in ['gender', 'stress_level', 'academic_work_impact', 'age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time']]
    print(f"Features: {len(feature_cols)} (added {len(agg_cols)} aggregation features)")
    print(f"Aggregation features: {agg_cols}")
    print(f"Train samples: {len(X)}")
    
    print("\nTraining LightGBM with 5-fold CV...")
    cv_auc, oof_preds = train_lgb_cv(X, y)
    
    results = pd.DataFrame({
        'exp_id': ['FE-06'],
        'category': ['Feature Eng'],
        'description': ['Aggregation features: total_screen, ratios, leisure/productive, sleep_debt, age_groups, intensity, weekend patterns'],
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
        'fe06_oof': oof_preds
    })
    oof_path = Path('/kaggle/working/fe06_oof.csv') if Path('/kaggle/working').exists() else Path('../models/fe06_oof.csv')
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to {oof_path}")

if __name__ == '__main__':
    main()