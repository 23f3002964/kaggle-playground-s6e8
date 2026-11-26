"""
Stacking Experiment: ST-03
LightGBM meta-learner on top model OOF predictions
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
    return X, y, X_test

def get_oof_predictions(X, y, model_name):
    """Get OOF predictions for a specific model"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    
    if model_name == 'lgb':
        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
            'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1,
            'seed': SEED, 'n_jobs': -1,
        }
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            model = lgb.train(params, train_data, num_boost_round=2000,
                            valid_sets=[train_data, val_data], valid_names=['train', 'valid'],
                            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    
    elif model_name == 'xgb':
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
            'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'min_child_weight': 1,
            'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': SEED,
            'n_jobs': -1, 'verbosity': 0,
        }
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            model = xgb.train(params, dtrain, num_boost_round=2000,
                            evals=[(dtrain, 'train'), (dval, 'valid')],
                            early_stopping_rounds=100, verbose_eval=False)
            oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    
    elif model_name == 'hgb':
        params = {
            'loss': 'log_loss', 'learning_rate': 0.05, 'max_iter': 2000,
            'max_depth': 6, 'min_samples_leaf': 20, 'l2_regularization': 1.0,
            'max_bins': 255, 'random_state': SEED, 'early_stopping': True,
            'validation_fraction': 0.1, 'n_iter_no_change': 100, 'tol': 1e-4,
        }
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = HistGradientBoostingClassifier(**params)
            model.fit(X_train, y_train)
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    
    return oof_preds

def train_lgb_meta_cv(oof_stack, y, model_names):
    """Train LightGBM meta-learner with CV"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_meta = np.zeros(len(y))
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 50,
        'verbose': -1,
        'seed': SEED,
        'n_jobs': -1,
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(oof_stack, y)):
        X_train, X_val = oof_stack[train_idx], oof_stack[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=model_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=model_names)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        oof_meta[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        fold_auc = roc_auc_score(y_val, oof_meta[val_idx])
        print(f"  Meta Fold {fold+1}: AUC = {fold_auc:.5f}")
    
    meta_auc = roc_auc_score(y, oof_meta)
    print(f"  LightGBM Meta-Learner CV AUC: {meta_auc:.5f}")
    return meta_auc, oof_meta

def main():
    print("="*60)
    print("ST-03: LightGBM Meta-Learner Stacking")
    print("="*60)
    
    train, test = load_data()
    print("Preparing features with FE-03 missingness indicators...")
    X, y, X_test = get_features(train, test)
    print(f"Features: {X.shape[1]}, Train samples: {len(X)}")
    
    # Get OOF predictions from top 3 models
    model_names = ['xgb', 'lgb', 'hgb']
    oof_dict = {}
    
    for name in model_names:
        print(f"\nTraining {name.upper()} for OOF...")
        oof_dict[name] = get_oof_predictions(X, y, name)
        auc = roc_auc_score(y, oof_dict[name])
        print(f"  {name} OOF AUC: {auc:.5f}")
    
    # Stack with LightGBM meta-learner
    print("\nTraining LightGBM meta-learner with CV...")
    oof_stack = np.column_stack([oof_dict[name] for name in model_names])
    
    meta_auc, meta_oof = train_lgb_meta_cv(oof_stack, y, model_names)
    
    # Compare with simple averages
    print("\nComparison:")
    for name in ['avg_top2', 'avg_top3']:
        if name == 'avg_top2':
            simple_avg = np.mean([oof_dict['xgb'], oof_dict['lgb']], axis=0)
        else:
            simple_avg = np.mean([oof_dict[m] for m in model_names], axis=0)
        auc = roc_auc_score(y, simple_avg)
        print(f"  {name}: {auc:.5f}")
    
    print(f"  LightGBM Meta: {meta_auc:.5f}")
    
    # Save results
    results = []
    for name in model_names:
        auc = roc_auc_score(y, oof_dict[name])
        results.append(('single', name, auc))
    results.append(('lgb_meta', 'lgb_meta_xgb+lgb+hgb', meta_auc))
    results.append(('avg_top2', 'xgb+lgb', roc_auc_score(y, np.mean([oof_dict['xgb'], oof_dict['lgb']], axis=0))))
    results.append(('avg_top3', 'xgb+lgb+hgb', roc_auc_score(y, np.mean([oof_dict[m] for m in model_names], axis=0))))
    
    results_df = pd.DataFrame(results, columns=['type', 'models', 'cv_auc'])
    results_df['exp_id'] = 'ST-03'
    results_df['category'] = 'Stacking'
    results_df['description'] = 'LightGBM meta-learner on top 3 model OOF predictions'
    results_df['status'] = 'completed'
    
    logs_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('../logs')
    logs_dir.mkdir(exist_ok=True)
    exp_file = logs_dir / 'experiments.csv'
    if exp_file.exists():
        existing = pd.read_csv(exp_file)
        results_df = pd.concat([existing, results_df], ignore_index=True)
    results_df.to_csv(exp_file, index=False)
    print(f"\nResults saved to {exp_file}")
    
    # Save meta-learner OOF
    meta_df = pd.DataFrame({
        'id': train['id'],
        'lgb_meta': meta_oof
    })
    for name in model_names:
        meta_df[name] = oof_dict[name]
    
    meta_path = logs_dir / 'st03_meta_oof.csv'
    meta_df.to_csv(meta_path, index=False)
    print(f"Meta-learner OOF saved to {meta_path}")

if __name__ == '__main__':
    main()