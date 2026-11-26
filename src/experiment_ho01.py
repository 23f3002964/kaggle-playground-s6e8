"""
Hyperopt Experiment: HO-01
Optuna joint optimization of XGBoost, LightGBM, HistGradientBoosting
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
import optuna

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
N_TRIALS = 50  # Adjust based on time budget

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

def objective_xgb(trial, X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        'gamma': trial.suggest_float('gamma', 1e-4, 10, log=True),
        'random_state': SEED,
        'n_jobs': -1,
        'verbosity': 0,
    }
    
    for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(params, dtrain, num_boost_round=2000,
                        evals=[(dtrain, 'train'), (dval, 'valid')],
                        early_stopping_rounds=100, verbose_eval=False)
        oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    
    return roc_auc_score(y, oof_preds)

def objective_lgb(trial, X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-4, 10, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, 10, log=True),
        'verbose': -1,
        'seed': SEED,
        'n_jobs': -1,
    }
    
    for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model = lgb.train(params, train_data, num_boost_round=2000,
                        valid_sets=[train_data, val_data], valid_names=['train', 'valid'],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    
    return roc_auc_score(y, oof_preds)

def objective_hgb(trial, X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    
    params = {
        'loss': 'log_loss',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_iter': 2000,
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 10, log=True),
        'max_bins': trial.suggest_int('max_bins', 63, 255),
        'random_state': SEED,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 100,
        'tol': 1e-4,
    }
    
    for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = HistGradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    
    return roc_auc_score(y, oof_preds)

def main():
    print("="*60)
    print("HO-01: Optuna Joint Optimization")
    print("="*60)
    
    train, test = load_data()
    print("Preparing features with FE-03 missingness indicators...")
    X, y = get_features(train, test)
    print(f"Features: {X.shape[1]}, Train: {len(X)}")
    
    results = []
    
    # Optimize XGBoost
    print("\n=== Optimizing XGBoost ===")
    study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_xgb.optimize(lambda t: objective_xgb(t, X, y), n_trials=N_TRIALS)
    best_xgb = study_xgb.best_value
    print(f"Best XGBoost AUC: {best_xgb:.5f}")
    print(f"Best params: {study_xgb.best_params}")
    results.append(('xgb_opt', best_xgb, study_xgb.best_params))
    
    # Optimize LightGBM
    print("\n=== Optimizing LightGBM ===")
    study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_lgb.optimize(lambda t: objective_lgb(t, X, y), n_trials=N_TRIALS)
    best_lgb = study_lgb.best_value
    print(f"Best LightGBM AUC: {best_lgb:.5f}")
    print(f"Best params: {study_lgb.best_params}")
    results.append(('lgb_opt', best_lgb, study_lgb.best_params))
    
    # Optimize HistGradientBoosting
    print("\n=== Optimizing HistGradientBoosting ===")
    study_hgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_hgb.optimize(lambda t: objective_hgb(t, X, y), n_trials=N_TRIALS)
    best_hgb = study_hgb.best_value
    print(f"Best HGB AUC: {best_hgb:.5f}")
    print(f"Best params: {study_hgb.best_params}")
    results.append(('hgb_opt', best_hgb, study_hgb.best_params))
    
    # Joint optimization (using best params from individual)
    print("\n=== Joint Optimization (Fine-tuning) ===")
    
    # Train final ensemble with best params
    # XGBoost final
    # ... (use best params from studies)
    
    # Save results
    results_df = pd.DataFrame([
        {'model': 'xgb_optimized', 'cv_auc': results[0][0], 'params': str(results[0][2])},
        {'model': 'lgb_optimized', 'cv_auc': results[1][0], 'params': str(results[1][2])},
        {'model': 'hgb_optimized', 'cv_auc': results[2][0], 'params': str(results[2][2])},
    ])
    results_df['exp_id'] = 'HO-01'
    results_df['category'] = 'Hyperopt'
    results_df['description'] = 'Optuna individual model optimization'
    results_df['status'] = 'completed'
    
    logs_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('../logs')
    logs_dir.mkdir(exist_ok=True)
    exp_file = logs_dir / 'experiments.csv'
    if exp_file.exists():
        existing = pd.read_csv(exp_file)
        results_df = pd.concat([existing, results_df], ignore_index=True)
    results_df.to_csv(exp_file, index=False)
    print(f"\nResults saved to {exp_file}")
    
    # Save study objects for later use
    import joblib
    joblib.dump(study_xgb, logs_dir / 'study_xgb.pkl')
    joblib.dump(study_lgb, logs_dir / 'study_lgb.pkl')
    joblib.dump(study_hgb, logs_dir / 'study_hgb.pkl')
    print(f"Studies saved to {logs_dir}")

if __name__ == '__main__':
    main()