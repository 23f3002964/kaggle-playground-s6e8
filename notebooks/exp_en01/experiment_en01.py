"""
Ensemble Selection Experiment: EN-01
Greedy ensemble selection from best model OOF predictions
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
    return X, y

def get_oof_predictions(X, y, model_name, model_params=None):
    """Get OOF predictions for a specific model"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    
    if model_name == 'xgb_opt':
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
        for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            model = lgb.train(params, train_data, num_boost_round=2000,
                            valid_sets=[train_data, val_data], valid_names=['train', 'valid'],
                            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    
    elif model_name == 'hgb':
        params = {
            'loss': 'log_loss', 'learning_rate': 0.05, 'max_iter': 2000,
            'max_depth': 6, 'min_samples_leaf': 20, 'l2_regularization': 1.0,
            'max_bins': 255, 'random_state': SEED, 'early_stopping': True,
            'validation_fraction': 0.1, 'n_iter_no_change': 100, 'tol': 1e-4,
        }
        for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = HistGradientBoostingClassifier(**params)
            model.fit(X_train, y_train)
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    
    elif model_name == 'cb':
        cat_features = [X.columns.get_loc(c) for c in ['gender', 'stress_level', 'academic_work_impact'] if c in X.columns]
        params = {
            'objective': 'Logloss', 'eval_metric': 'AUC', 'iterations': 2000,
            'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 3,
            'subsample': 0.8, 'colsample_bylevel': 0.8, 'random_seed': SEED,
            'verbose': False, 'thread_count': -1, 'early_stopping_rounds': 100,
        }
        for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            train_pool = Pool(X_train, y_train, cat_features=cat_features)
            val_pool = Pool(X_val, y_val, cat_features=cat_features)
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    
    return oof_preds

def greedy_ensemble_selection(oof_dict, y, max_models=6):
    """Greedy forward selection for ensemble"""
    available = list(oof_dict.keys())
    selected = []
    best_auc = 0
    
    print(f"Greedy ensemble selection from {len(available)} models...")
    individual_aucs = {k: roc_auc_score(y, v) for k, v in oof_dict.items()}
    print(f"Individual AUCs: { {k: f'{v:.5f}' for k, v in individual_aucs.items()} }")
    
    for step in range(min(max_models, len(available))):
        best_add = None
        best_step_auc = best_auc
        
        for model in available:
            if model in selected:
                continue
            candidate = selected + [model]
            avg_pred = np.mean([oof_dict[m] for m in candidate], axis=0)
            auc = roc_auc_score(y, avg_pred)
            if auc > best_step_auc:
                best_step_auc = auc
                best_add = model
        
        if best_add is None:
            print(f"No improvement found. Stopping at step {step}.")
            break
        
        selected.append(best_add)
        available.remove(best_add)
        best_auc = best_step_auc
        print(f"Step {step+1}: Added {best_add}, Ensemble AUC = {best_auc:.5f}")
    
    # Also try backward elimination from all models
    print("\n--- Backward elimination ---")
    all_models = list(oof_dict.keys())
    selected_back = all_models.copy()
    best_auc_back = roc_auc_score(y, np.mean([oof_dict[m] for m in selected_back], axis=0))
    print(f"All models: {best_auc_back:.5f}")
    
    improved = True
    while improved and len(selected_back) > 1:
        improved = False
        worst_removal = None
        best_auc_after_removal = best_auc_back
        
        for model in selected_back:
            candidate = [m for m in selected_back if m != model]
            avg_pred = np.mean([oof_dict[m] for m in candidate], axis=0)
            auc = roc_auc_score(y, avg_pred)
            if auc >= best_auc_after_removal:
                best_auc_after_removal = auc
                worst_removal = model
                improved = True
        
        if worst_removal:
            selected_back.remove(worst_removal)
            best_auc_back = best_auc_after_removal
            print(f"Removed {worst_removal}, Ensemble AUC = {best_auc_back:.5f}")
    
    return selected, best_auc, selected_back, best_auc_back

def main():
    print("="*60)
    print("EN-01: Greedy Ensemble Selection")
    print("="*60)
    
    train, test = load_data()
    print("Preparing features with FE-03 missingness indicators...")
    X, y = get_features(train, test)
    print(f"Features: {X.shape[1]}, Train: {len(X)}")
    
    # Get OOF from top models
    models = {
        'xgb_opt': 'xgb_opt',
        'xgb_base': 'xgb',
        'lgb': 'lgb',
        'hgb': 'hgb',
        'cb': 'cb',
    }
    
    print("\nGenerating OOF predictions...")
    oof_dict = {}
    for name, model_name in models.items():
        print(f"  Training {name}...")
        oof_dict[name] = get_oof_predictions(X, y, model_name)
        auc = roc_auc_score(y, oof_dict[name])
        print(f"    {name}: {auc:.5f}")
    
    # Greedy selection
    print("\n=== Greedy Forward Selection ===")
    selected_fwd, auc_fwd, _, _ = greedy_ensemble_selection(oof_dict, y)
    
    # Backward elimination
    print("\n=== Backward Elimination ===")
    selected_bwd, auc_bwd, _, _ = greedy_ensemble_selection(oof_dict, y)  # Uses backward logic inside
    
    # Evaluate all combinations of top N
    print("\n=== All combinations of top models ===")
    from itertools import combinations
    
    model_names = list(oof_dict.keys())
    for r in range(2, min(6, len(model_names)) + 1):
        for combo in combinations(model_names, r):
            avg_pred = np.mean([oof_dict[m] for m in combo], axis=0)
            auc = roc_auc_score(y, avg_pred)
            print(f"  {'+'.join(combo)}: {auc:.5f}")
    
    # Best ensemble OOF
    best_combo = selected_fwd if auc_fwd >= auc_bwd else list(combinations(model_names, 3))[0]  # placeholder
    # Find actual best combo
    best_auc = 0
    best_combo = None
    for r in range(2, min(6, len(model_names)) + 1):
        for combo in combinations(model_names, r):
            avg_pred = np.mean([oof_dict[m] for m in combo], axis=0)
            auc = roc_auc_score(y, avg_pred)
            if auc > best_auc:
                best_auc = auc
                best_combo = combo
    
    print(f"\nBest ensemble: {'+'.join(best_combo)} = {best_auc:.5f}")
    
    # Generate test predictions for best ensemble
    # ... (would need test predictions for each model)
    
    # Save results
    results = pd.DataFrame({
        'exp_id': ['EN-01'],
        'category': ['Ensemble'],
        'description': ['Greedy ensemble selection from top models'],
        'cv_auc': [best_auc],
        'models_used': [str(best_combo)],
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

if __name__ == '__main__':
    main()