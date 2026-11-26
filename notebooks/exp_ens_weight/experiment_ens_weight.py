"""
Ensemble Weight Optimization
Find optimal weights for model ensemble using OOF predictions
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
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
    return X, y, X_test

def get_oof(model_name, X, y, params=None):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    
    if model_name == 'xgb':
        params_xgb = {
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
            'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'min_child_weight': 1,
            'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': SEED,
            'n_jobs': -1, 'verbosity': 0,
        }
        p = {**params_xgb, **(params or {})}
        for fold, (tr, val) in enumerate(skf.split(X, y)):
            dtrain = xgb.DMatrix(X.iloc[tr], label=y.iloc[tr])
            dval = xgb.DMatrix(X.iloc[val], label=y.iloc[val])
            m = xgb.train(p, dtrain, 2000, [(dtrain, 'tr'), (dval, 'val')],
                         early_stopping_rounds=100, verbose_eval=0)
            oof[val] = m.predict(dval, iteration_range=(0, m.best_iteration+1))
    
    elif model_name == 'lgb':
        params_lgb = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
            'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1,
            'seed': SEED, 'n_jobs': -1,
        }
        p = {**params_lgb, **(params or {})}
        for fold, (tr, val) in enumerate(skf.split(X, y)):
            tr_data = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
            val_data = lgb.Dataset(X.iloc[val], label=y.iloc[val], reference=tr_data)
            callbacks = [lgb.early_stopping(100), lgb.log_evaluation(0)]
            m = lgb.train(p, tr_data, 2000, [tr_data, val_data], ['tr', 'val'], callbacks=callbacks)
            oof[val] = m.predict(X.iloc[val], num_iteration=m.best_iteration)
    
    elif model_name == 'hgb':
        params_hgb = {
            'loss': 'log_loss', 'learning_rate': 0.05, 'max_iter': 2000,
            'max_depth': 6, 'min_samples_leaf': 20, 'l2_regularization': 1.0,
            'max_bins': 255, 'random_state': SEED, 'early_stopping': True,
            'validation_fraction': 0.1, 'n_iter_no_change': 100, 'tol': 1e-4,
        }
        p = {**params_hgb, **(params or {})}
        for fold, (tr, val) in enumerate(skf.split(X, y)):
            m = HistGradientBoostingClassifier(**params_hgb)
            m.fit(X.iloc[tr], y.iloc[tr])
            oof[val] = m.predict_proba(X.iloc[val])[:, 1]
    
    elif model_name == 'cb':
        cat_features = [X.columns.get_loc(c) for c in ['gender', 'stress_level', 'academic_work_impact'] if c in X.columns]
        params_cb = {
            'objective': 'Logloss', 'eval_metric': 'AUC', 'iterations': 2000,
            'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 3,
            'subsample': 0.8, 'colsample_bylevel': 0.8, 'random_seed': SEED,
            'verbose': False, 'thread_count': -1, 'early_stopping_rounds': 100,
        }
        for fold, (tr, val) in enumerate(skf.split(X, y)):
            train_pool = Pool(X.iloc[tr], y.iloc[tr], cat_features=cat_features)
            val_pool = Pool(X.iloc[val], y.iloc[val], cat_features=cat_features)
            m = CatBoostClassifier(**params_cb)
            m.fit(train_pool, eval_set=val_pool, use_best_model=True)
            oof[val] = m.predict_proba(X.iloc[val])[:, 1]
    
    return oof

def optimize_weights(oof_dict, y):
    """Find optimal weights for ensemble using gradient-based optimization"""
    model_names = list(oof_dict.keys())
    oof_matrix = np.column_stack([oof_dict[m] for m in oof_dict])
    
    def objective(weights):
        # Softmax to ensure positive weights summing to 1
        w = np.exp(weights) / np.sum(np.exp(weights))
        ensemble = oof_matrix @ w
        return -roc_auc_score(y, ensemble)
    
    # Initial guess: uniform weights
    init_weights = np.zeros(len(oof_dict))
    
    result = minimize(objective, init_weights, method='L-BFGS-B', 
                     options={'maxiter': 1000, 'ftol': 1e-10})
    
    best_weights = np.exp(result.x) / np.sum(np.exp(result.x))
    best_auc = -result.fun
    
    return best_weights, best_auc

def main():
    print("="*60)
    print("Ensemble Weight Optimization")
    print("="*60)
    
    train, test = load_data()
    print("Preparing features with FE-03 missingness indicators...")
    X, y, X_test = get_features(train, test)
    print(f"Features: {X.shape[1]}, Train: {len(X)}")
    
    # Get OOF from top models
    models = ['xgb', 'lgb', 'hgb', 'cb']
    oof_dict = {}
    
    for name in models:
        print(f"\nTraining {name}...")
        oof_dict[name] = get_oof(name, X, y)
        auc = roc_auc_score(y, oof_dict[name])
        print(f"  {name} OOF AUC: {auc:.5f}")
    
    # Optimize weights
    print("\n=== Optimizing Ensemble Weights ===")
    best_weights, best_auc = optimize_weights(oof_dict, y)
    
    print(f"\nOptimal Weights:")
    for name, w in zip(oof_dict.keys(), best_weights):
        print(f"  {name}: {w:.4f}")
    print(f"Optimized AUC: {best_auc:.5f}")
    
    # Compare with uniform
    uniform_auc = roc_auc_score(y, np.mean([oof_dict[m] for m in oof_dict], axis=0))
    print(f"Uniform average: {uniform_auc:.5f}")
    
    # Try different regularization strengths
    print("\n=== L2 Regularization Sweep ===")
    from scipy.optimize import minimize
    
    def objective_reg(weights, reg):
        w = np.exp(weights) / np.sum(np.exp(weights))
        ensemble = oof_matrix @ w
        return -roc_auc_score(y, ensemble) + reg * np.sum(w**2)
    
    oof_matrix = np.column_stack([oof_dict[m] for m in ['xgb', 'lgb', 'hgb', 'cb']])
    
    for reg in [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        init = np.zeros(4)
        res = minimize(lambda w: -roc_auc_score(y, oof_matrix @ (np.exp(w)/np.sum(np.exp(w)))) + reg * np.sum((np.exp(w)/np.sum(np.exp(w)))**2), 
                      np.zeros(4), method='L-BFGS-B', options={'maxiter': 500})
        w = np.exp(res.x) / np.sum(np.exp(res.x))
        auc = roc_auc_score(y, oof_matrix @ w)
        print(f"  reg={reg}: AUC={auc:.5f}, weights={w}")
    
    # Save results
    results = []
    results.append(('opt_weights', best_auc))
    results.append(('uniform', roc_auc_score(y, np.mean([oof_dict[m] for m in oof_dict], axis=0))))
    
    results_df = pd.DataFrame(results, columns=['method', 'cv_auc'])
    results_df['exp_id'] = 'ENS_WEIGHT'
    results_df['category'] = 'Ensemble Weight Optimization'
    results_df['description'] = 'Optimize ensemble weights using gradient-based optimization'
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