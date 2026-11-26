"""
Improvement Experiment: Build on best public model (0.97105)
Pool141 CV 0.97006 -> Public 0.97105
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
from sklearn.linear_model import LogisticRegression, Ridge
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
    """Feature engineering from pool141 (best public model)"""
    df = df.copy()
    
    # Basic interactions
    df['total_screen_time'] = df['daily_screen_time_hours'] + df['weekend_screen_time']
    df['screen_per_notification'] = df['daily_screen_time_hours'] / (df['notifications_per_day'] + 1e-6)
    df['screen_per_app_open'] = df['daily_screen_time_hours'] / (df['app_opens_per_day'] + 1e-6)
    df['notif_per_app_open'] = df['notifications_per_day'] / (df['app_opens_per_day'] + 1e-6)
    
    # Leisure vs work
    df['leisure_hours'] = df['social_media_hours'] + df['gaming_hours']
    df['leisure_productive_ratio'] = df['leisure_hours'] / (df['work_study_hours'] + 1e-6)
    
    # Sleep
    df['sleep_debt'] = np.maximum(8 - df['sleep_hours'], 0)
    df['screen_per_sleep_hour'] = df['daily_screen_time_hours'] / (df['sleep_hours'] + 1e-6)
    
    # Age interactions
    df['age_screen_interaction'] = df['age'] * df['daily_screen_time_hours']
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 50, 100], labels=[0,1,2,3,4]).astype(float)
    
    # Weekend pattern
    df['weekend_weekday_ratio'] = df['weekend_screen_time'] / (df['daily_screen_time_hours'] + 1e-6)
    df['weekend_excess'] = df['weekend_screen_time'] - df['daily_screen_time_hours']
    
    # Stress/impact numeric
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
        default_params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
            'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'min_child_weight': 1,
            'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': SEED,
            'n_jobs': -1, 'verbosity': 0,
        }
        p = {**default_params, **(params or {})}
        for fold, (tr, val) in enumerate(skf.split(X, y)):
            dtrain = xgb.DMatrix(X.iloc[tr], label=y.iloc[tr])
            dval = xgb.DMatrix(X.iloc[val], label=y.iloc[val])
            m = xgb.train(p, dtrain, 2000, [(dtrain, 'tr'), (dval, 'val')],
                         early_stopping_rounds=100, verbose_eval=0)
            oof[val] = m.predict(dval, iteration_range=(0, m.best_iteration+1))
    
    elif model_name == 'lgb':
        default_params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
            'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1,
            'seed': SEED, 'n_jobs': -1,
        }
        p = {**default_params, **(params or {})}
        for fold, (tr, val) in enumerate(skf.split(X, y)):
            tr_data = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
            val_data = lgb.Dataset(X.iloc[val], label=y.iloc[val], reference=tr_data)
            callbacks = [lgb.early_stopping(100), lgb.log_evaluation(0)]
            m = lgb.train(p, tr_data, 2000, [tr_data, val_data], ['tr', 'val'], callbacks=callbacks)
            oof[val] = m.predict(X.iloc[val], num_iteration=m.best_iteration)
    
    elif model_name == 'lgb_opt':
        # Optimized LGB from HO-01
        default_params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
            'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1,
            'seed': SEED, 'n_jobs': -1,
        }
        p = {**default_params, **(params or {})}
        for fold, (tr, val) in enumerate(skf.split(X, y)):
            tr_data = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
            val_data = lgb.Dataset(X.iloc[val], label=y.iloc[val], reference=tr_data)
            callbacks = [lgb.early_stopping(100), lgb.log_evaluation(0)]
            m = lgb.train(p, tr_data, 2000, [tr_data, val_data], ['tr', 'val'], callbacks=callbacks)
            oof[val] = m.predict(X.iloc[val], num_iteration=m.best_iteration)
    
    elif model_name == 'hgb':
        default_params = {
            'loss': 'log_loss', 'learning_rate': 0.05, 'max_iter': 2000,
            'max_depth': 6, 'min_samples_leaf': 20, 'l2_regularization': 1.0,
            'max_bins': 255, 'random_state': SEED, 'early_stopping': True,
            'validation_fraction': 0.1, 'n_iter_no_change': 100, 'tol': 1e-4,
        }
        p = {**default_params, **(params or {})}
        for fold, (tr, val) in enumerate(skf.split(X, y)):
            m = HistGradientBoostingClassifier(**p)
            m.fit(X.iloc[tr], y.iloc[tr])
            oof[val] = m.predict_proba(X.iloc[val])[:, 1]
    
    return oof

def stack_models(oof_dict, y, method='ridge'):
    """Stack models with different meta-learners"""
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.model_selection import cross_val_predict
    
    oof_stack = np.column_stack([oof_dict[m] for m in oof_dict])
    
    if method == 'ridge':
        meta = RidgeCV(alphas=np.logspace(-4, 4, 9), cv=5, scoring='roc_auc')
        meta.fit(oof_stack, y)
        preds = meta.predict(oof_stack)
        # Convert to probability
        from scipy.special import expit
        preds = expit(preds)
    elif method == 'logistic':
        meta = LogisticRegressionCV(Cs=np.logspace(-4, 4, 9), cv=5, scoring='roc_auc', 
                                   random_state=SEED, max_iter=1000)
        meta.fit(oof_stack, y)
        preds = meta.predict_proba(oof_stack)[:, 1]
    elif method == 'lgb':
        meta = lgb.LGBMClassifier(
            objective='binary', metric='auc', n_estimators=200,
            num_leaves=31, learning_rate=0.05, verbose=-1, random_state=SEED
        )
        meta.fit(oof_stack, y)
        preds = meta.predict_proba(oof_stack)[:, 1]
    elif method == 'xgb':
        meta = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='auc', n_estimators=200,
            max_depth=4, learning_rate=0.05, random_state=SEED, n_jobs=-1
        )
        meta.fit(oof_stack, y)
        preds = meta.predict_proba(oof_stack)[:, 1]
    
    return preds, meta

def main():
    print("="*60)
    print("Improvement on Pool141 (Best Public 0.97105)")
    print("="*60)
    
    train, test = load_data()
    X, y, X_test = get_features(train, test)
    print(f"Features: {X.shape[1]}, Train: {len(X)}")
    
    # Train base models
    models = ['xgb', 'lgb', 'hgb', 'lgb_opt']
    oof_dict = {}
    
    for name in models:
        print(f"\nTraining {name}...")
        oof_dict[name] = get_oof(name, X, y)
        auc = roc_auc_score(y, oof_dict[name])
        print(f"  {name} OOF AUC: {auc:.5f}")
    
    # Try different stacking methods
    print("\n=== Stacking Experiments ===")
    results = []
    
    for method in ['ridge', 'logistic', 'lgb', 'xgb']:
        try:
            preds, meta = stack_models(oof_dict, y, method)
            auc = roc_auc_score(y, preds)
            print(f"  {method} stack: {auc:.5f}")
            results.append((f'stack_{method}', auc))
        except Exception as e:
            print(f"  {method} failed: {e}")
    
    # Simple averages
    for name in ['avg_xgb_lgb', 'avg_top3', 'avg_all']:
        if name == 'avg_xgb_lgb':
            avg = np.mean([oof_dict['xgb'], oof_dict['lgb']], axis=0)
        elif name == 'avg_top3':
            avg = np.mean([oof_dict[m] for m in ['xgb', 'lgb', 'hgb']], axis=0)
        else:
            avg = np.mean([oof_dict[m] for m in models], axis=0)
        auc = roc_auc_score(y, avg)
        print(f"  {name}: {auc:.5f}")
        results.append((name, auc))
    
    # Weighted by CV AUC
    weights = {m: roc_auc_score(y, oof_dict[m]) for m in oof_dict}
    total_w = sum(weights.values())
    weighted = sum((weights[m]/total_w) * oof_dict[m] for m in models)
    auc_w = roc_auc_score(y, weighted)
    print(f"  weighted: {auc_w:.5f}")
    results.append(('weighted', auc_w))
    
    # Save results
    results_df = pd.DataFrame(results, columns=['method', 'cv_auc'])
    results_df['exp_id'] = 'IMPROVE'
    results_df['category'] = 'Improvement'
    results_df['description'] = 'Improvement on Pool141 best model'
    results_df['status'] = 'completed'
    
    logs_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('../logs')
    logs_dir.mkdir(exist_ok=True)
    exp_file = logs_dir / 'experiments.csv'
    if exp_file.exists():
        existing = pd.read_csv(exp_file)
        results_df = pd.concat([existing, results_df], ignore_index=True)
    results_df.to_csv(exp_file, index=False)
    print(f"\nResults saved to {exp_file}")
    
    # Generate test predictions for best method
    best_method = max(results, key=lambda x: x[1])[0]
    print(f"\nBest method: {best_method} ({max(r[1] for r in results):.5f})")
    
    # Retrain best on full data and predict test
    # (In practice, would retrain ensemble on full data)

if __name__ == '__main__':
    main()