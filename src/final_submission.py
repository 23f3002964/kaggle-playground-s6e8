"""
Final Submission Generation
Best Ensemble: xgb_opt + lgb = 0.96425
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
    X_test = test[feature_cols]
    return X, y, X_test, feature_cols

def train_xgb_opt(X, y, X_test):
    """Train optimized XGBoost and return test predictions"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    test_preds = np.zeros(len(X_test))
    
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
        'max_depth': 5, 'learning_rate': 0.09891017582644358, 'subsample': 0.9721476556449753,
        'colsample_bytree': 0.6625321775589934, 'min_child_weight': 8,
        'reg_alpha': 5.899522236163052, 'reg_lambda': 0.199614235591803,
        'gamma': 0.015390637444741026, 'random_state': SEED, 'n_jobs': -1, 'verbosity': 0,
    }
    
    for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)
        model = xgb.train(params, dtrain, num_boost_round=2000,
                        evals=[(dtrain, 'train'), (dval, 'valid')],
                        early_stopping_rounds=100, verbose_eval=0)
        test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / N_FOLDS
    
    return test_preds

def train_lgb(X, y, X_test):
    """Train LightGBM and return test predictions"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    test_preds = np.zeros(len(X_test))
    
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1,
        'seed': SEED, 'n_jobs': -1,
    }
    
    for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model = lgb.train(params, train_data, num_boost_round=2000,
                        valid_sets=[train_data, val_data], valid_names=['train', 'valid'],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
    
    return test_preds

def main():
    print("="*60)
    print("Final Submission Generation: xgb_opt + lgb ensemble")
    print("="*60)
    
    train, test = load_data()
    print("Preparing features with FE-03 missingness indicators...")
    X, y, X_test, feature_cols = get_features(train, test)
    print(f"Features: {X.shape[1]}, Train: {len(X)}, Test: {len(X_test)}")
    
    # Train both models and get test predictions
    print("\nTraining optimized XGBoost...")
    xgb_test = train_xgb_opt(X, y, X_test)
    
    print("Training LightGBM...")
    lgb_test = train_lgb(X, y, X_test)
    
    # Ensemble: average of xgb_opt + lgb, clipped to [0, 1]
    ensemble_test = np.clip((xgb_test + lgb_test) / 2, 0, 1)
    
    # Generate submission
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    submission = pd.DataFrame({
        'id': sample_sub['id'],
        'addicted_label': ensemble_test
    })
    
    # Save submission
    sub_path = Path('/kaggle/working/submission_ensemble_0.96425.csv') if Path('/kaggle/working').exists() else Path('../submissions/submission_ensemble_0.96425.csv')
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(sub_path, index=False)
    print(f"\nSubmission saved to {sub_path}")
    print(f"Shape: {submission.shape}")
    print(f"Mean prediction: {ensemble_test.mean():.5f}")
    print(f"Min prediction: {ensemble_test.min():.5f}")
    print(f"Max prediction: {ensemble_test.max():.5f}")
    
    # Also save individual model predictions for reference
    xgb_sub = pd.DataFrame({'id': sample_sub['id'], 'addicted_label': xgb_test})
    lgb_sub = pd.DataFrame({'id': sample_sub['id'], 'addicted_label': lgb_test})
    
    Path('/kaggle/working').mkdir(parents=True, exist_ok=True)
    xgb_sub.to_csv('/kaggle/working/submission_xgb_opt.csv', index=False)
    lgb_sub.to_csv('/kaggle/working/submission_lgb.csv', index=False)
    print("Individual model submissions saved")

if __name__ == '__main__':
    main()