"""
CV Strategy Experiment: CV-03
Pseudo-labeling: Add high-confidence test predictions to training data
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
    return X, y, X_test, feature_cols

def train_ensemble_oof(X, y, X_test, model_names):
    """Train multiple models and get OOF + test predictions"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    oof_preds = {name: np.zeros(len(X)) for name in model_names}
    test_preds = {name: np.zeros(len(X_test)) for name in model_names}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # XGBoost
        if 'xgb' in model_names:
            params = {
                'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
                'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
                'colsample_bytree': 0.8, 'min_child_weight': 1,
                'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': SEED,
                'n_jobs': -1, 'verbosity': 0,
            }
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            dtest = xgb.DMatrix(X_test)
            model = xgb.train(params, dtrain, num_boost_round=2000,
                            evals=[(dtrain, 'train'), (dval, 'valid')],
                            early_stopping_rounds=100, verbose_eval=False)
            oof_preds['xgb'][val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
            test_preds['xgb'] += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / N_FOLDS
        
        # LightGBM
        if 'lgb' in model_names:
            params = {
                'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
                'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
                'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1,
                'seed': SEED, 'n_jobs': -1,
            }
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            model = lgb.train(params, train_data, num_boost_round=2000,
                            valid_sets=[train_data, val_data], valid_names=['train', 'valid'],
                            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            oof_preds['lgb'][val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
            test_preds['lgb'] += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
        
        # HistGradientBoosting
        if 'hgb' in model_names:
            from sklearn.ensemble import HistGradientBoostingClassifier
            params = {
                'loss': 'log_loss', 'learning_rate': 0.05, 'max_iter': 2000,
                'max_depth': 6, 'min_samples_leaf': 20, 'l2_regularization': 1.0,
                'max_bins': 255, 'random_state': SEED, 'early_stopping': True,
                'validation_fraction': 0.1, 'n_iter_no_change': 100, 'tol': 1e-4,
            }
            model = HistGradientBoostingClassifier(**params)
            model.fit(X_train, y_train)
            oof_preds['hgb'][val_idx] = model.predict_proba(X_val)[:, 1]
            test_preds['hgb'] += model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    return oof_preds, test_preds

def run_pseudo_labeling(train, test, X, y, X_test, feature_cols, oof_preds, test_preds, 
                       high_thresh=0.95, low_thresh=0.05, model_name='xgb'):
    """Add pseudo-labeled test data to training and re-evaluate"""
    
    # Average ensemble test predictions
    ensemble_test = np.mean([test_preds[m] for m in ['xgb', 'lgb', 'hgb']], axis=0)
    
    # Select high-confidence pseudo-labels
    high_conf = (ensemble_test >= high_thresh) | (ensemble_test <= low_thresh)
    n_pseudo = high_conf.sum()
    
    print(f"\nPseudo-labeling: {n_pseudo} high-confidence samples ({high_thresh}/{low_thresh})")
    
    if n_pseudo == 0:
        print("No high-confidence samples found")
        return None
    
    # Create pseudo-labeled dataset
    pseudo_labels = (ensemble_test[high_conf] >= 0.5).astype(int)
    pseudo_X = X_test[high_conf].copy()
    pseudo_y = pd.Series(pseudo_labels, index=pseudo_X.index)
    
    # Combine with original
    X_combined = pd.concat([X, pseudo_X], axis=0)
    y_combined = pd.concat([y, pseudo_y], axis=0)
    
    print(f"Combined training size: {len(X_combined)} (original: {len(X)} + pseudo: {n_pseudo})")
    
    # Evaluate with CV on combined data
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X_combined))
    
    if model_name == 'xgb':
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
            'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'min_child_weight': 1,
            'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': SEED,
            'n_jobs': -1, 'verbosity': 0,
        }
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined)):
            X_train, X_val = X_combined.iloc[train_idx], X_combined.iloc[val_idx]
            y_train, y_val = y_combined.iloc[train_idx], y_combined.iloc[val_idx]
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            model = xgb.train(params, dtrain, num_boost_round=2000,
                            evals=[(dtrain, 'train'), (dval, 'valid')],
                            early_stopping_rounds=100, verbose_eval=False)
            
            oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    
    # Only evaluate on original training data (not pseudo)
    original_auc = roc_auc_score(y, oof_preds[:len(y)])
    print(f"  Original data CV AUC with pseudo-labeling: {original_auc:.5f}")
    
    return original_auc, oof_preds[:len(y)]

def main():
    print("="*60)
    print("CV-03: Pseudo-Labeling Experiment")
    print("="*60)
    
    train, test = load_data()
    print("Preparing features with FE-03 missingness indicators...")
    X, y, X_test, feature_cols = get_features(train, test)
    print(f"Features: {X.shape[1]}, Train: {len(X)}, Test: {len(X_test)}")
    
    model_names = ['xgb', 'lgb', 'hgb']
    
    # Train ensemble and get OOF + test predictions
    print("\nTraining ensemble for OOF and test predictions...")
    oof_preds, test_preds = train_ensemble_oof(X, y, X_test, model_names)
    
    # Evaluate individual OOF
    for name in model_names:
        auc = roc_auc_score(y, oof_preds[name])
        print(f"  {name} OOF AUC: {auc:.5f}")
    
    # Baseline ensemble
    ensemble_oof = np.mean([oof_preds[m] for m in model_names], axis=0)
    baseline_auc = roc_auc_score(y, ensemble_oof)
    print(f"  Ensemble OOF AUC: {baseline_auc:.5f}")
    
    # Try pseudo-labeling with different thresholds
    results = [('baseline', 'ensemble', baseline_auc)]
    
    for high_thresh, low_thresh in [(0.95, 0.05), (0.97, 0.03), (0.99, 0.01), (0.90, 0.10)]:
        print(f"\n--- Pseudo-labeling: high={high_thresh}, low={low_thresh} ---")
        for model_name in ['xgb', 'lgb']:
            try:
                auc, _ = run_pseudo_labeling(train, test, X, y, X_test, feature_cols, 
                                            oof_preds, test_preds, 
                                            high_thresh, low_thresh, model_name)
                if auc:
                    results.append((f'pseudo_{high_thresh}_{low_thresh}', model_name, auc))
            except Exception as e:
                print(f"  Error with {model_name}: {e}")
    
    # Save results
    results_df = pd.DataFrame(results, columns=['type', 'model', 'cv_auc'])
    results_df['exp_id'] = 'CV-03'
    results_df['category'] = 'CV Strategy'
    results_df['description'] = 'Pseudo-labeling with high-confidence test predictions'
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