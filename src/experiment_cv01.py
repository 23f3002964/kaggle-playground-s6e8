"""
CV Strategy Experiment: CV-01
GroupKFold by age bins to prevent age leakage
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupKFold, StratifiedKFold
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
    df = df.copy()
    for col in num_cols:
        df[f'{col}_was_missing'] = df[col].isna().astype(int)
    return df

def get_features(train, test):
    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['id', 'addicted_label']]
    train = add_missingness_indicators(train, num_cols)
    test = add_missingness_indicators(test, num_cols)
    train = preprocess_basic(train)
    test = preprocess_basic(test)
    feature_cols = [c for c in train.columns if c not in ['id', 'addicted_label']]
    X = train[feature_cols]
    y = train['addicted_label']
    return X, y

def create_age_groups(age_series, n_bins=10):
    """Create age groups for GroupKFold"""
    # Fill missing ages with median first
    age_filled = age_series.fillna(age_series.median())
    # Use quantile bins to have roughly equal group sizes
    groups = pd.qcut(age_filled, q=n_bins, labels=False, duplicates='drop')
    if groups.isna().any():
        # Fallback to regular cut for NaN values
        groups = pd.cut(age_filled, bins=n_bins, labels=False)
    return groups.astype(int)

def train_model_cv(X, y, groups, model_name, cv_type='group'):
    """Train model with GroupKFold or StratifiedKFold"""
    if cv_type == 'group':
        cv = GroupKFold(n_splits=N_FOLDS)
        split_iter = cv.split(X, y, groups)
    else:
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        split_iter = cv.split(X, y)
    
    oof_preds = np.zeros(len(X))
    
    if model_name == 'xgb':
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
            'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'min_child_weight': 1,
            'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': SEED,
            'n_jobs': -1, 'verbosity': 0,
        }
        for fold, (train_idx, val_idx) in enumerate(split_iter):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            model = xgb.train(params, dtrain, num_boost_round=2000,
                            evals=[(dtrain, 'train'), (dval, 'valid')],
                            early_stopping_rounds=100, verbose_eval=False)
            oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    
    elif model_name == 'lgb':
        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
            'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1,
            'seed': SEED, 'n_jobs': -1,
        }
        for fold, (train_idx, val_idx) in enumerate(split_iter):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            model = lgb.train(params, train_data, num_boost_round=2000,
                            valid_sets=[train_data, val_data], valid_names=['train', 'valid'],
                            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    
    return oof_preds

def main():
    print("="*60)
    print("CV-01: GroupKFold by Age Bins")
    print("="*60)
    
    train, test = load_data()
    print("Preparing features with FE-03 missingness indicators...")
    X, y = get_features(train, test)
    
    # Create age groups
    age_groups = create_age_groups(train['age'], n_bins=10)
    print(f"Age groups: {age_groups.nunique()} unique groups")
    print(f"Group sizes: {age_groups.value_counts().sort_index().values}")
    
    model_names = ['xgb', 'lgb']
    cv_types = ['group', 'stratified']
    
    results = []
    
    for model_name in model_names:
        for cv_type in cv_types:
            print(f"\nTraining {model_name.upper()} with {cv_type} KFold...")
            oof_preds = train_model_cv(X, y, age_groups, model_name, cv_type)
            auc = roc_auc_score(y, oof_preds)
            print(f"  {model_name} {cv_type} AUC: {auc:.5f}")
            results.append({
                'type': f'{model_name}_{cv_type}',
                'model': model_name,
                'cv_type': cv_type,
                'cv_auc': auc
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df['exp_id'] = 'CV-01'
    results_df['category'] = 'CV Strategy'
    results_df['description'] = 'GroupKFold by age bins vs StratifiedKFold'
    results_df['status'] = 'completed'
    
    logs_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('../logs')
    logs_dir.mkdir(exist_ok=True)
    exp_file = logs_dir / 'experiments.csv'
    if exp_file.exists():
        existing = pd.read_csv(exp_file)
        results_df = pd.concat([existing, results_df], ignore_index=True)
    results_df.to_csv(exp_file, index=False)
    print(f"\nResults saved to {exp_file}")

if __name__ == '__main__':
    main()