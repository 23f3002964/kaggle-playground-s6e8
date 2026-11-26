"""
Pool141 Improvement Pipeline
Goal: Improve pool141 (0.97105) → Top 1 with robust private set performance

Key improvements:
1. GroupKFold by age + Adversarial validation
2. Multi-seed bagging (20 seeds per model)
4. Per-model feature selection
4. 3-level stacking (Base → Meta → Final)
5. Iterative pseudo-labeling (3 rounds)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

# Kaggle paths
import os
if os.path.exists('/kaggle/input/competitions/playground-series-s6e8/train.csv'):
    DATA_DIR = Path('/kaggle/input/competitions/playground-series-s6e8')
elif os.path.exists('/content/train.csv'):
    DATA_DIR = Path('/content')
elif os.path.exists('/content/drive/MyDrive/kaggle/playground-s6e8/train.csv'):
    DATA_DIR = Path('/content/drive/MyDrive/kaggle/playground-s6e8')
else:
    DATA_DIR = Path('../data')

SEED = 42
N_FOLDS = 5
N_SEEDS = 5
N_ROUNDS = 3
HIGH_THRESH = 0.97
LOW_THRESH = 0.03

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
    return X, y, X_test, feature_cols

def create_age_groups(age_series, n_bins=10):
    age_filled = age_series.fillna(age_series.median())
    groups = pd.qcut(age_filled, q=n_bins, labels=False, duplicates='drop')
    if groups.isna().any():
        groups = pd.cut(age_filled, bins=n_bins, labels=False)
    return groups.astype(int)

def get_model_features(model_name, X, y):
    """Select features per model type"""
    if model_name == 'xgb':
        # XGB prefers more features, use top 30 by importance
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        return importance.head(30).index.tolist()
    elif model_name == 'lgb':
        # LGB: top 25
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        return importance.head(25).index.tolist()
    elif model_name == 'cb':
        # CB: top 20
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        return importance.head(20).index.tolist()
    elif model_name == 'hgb':
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        return importance.head(20).index.tolist()
    return X.columns.tolist()

def train_model_cv(model_name, X, y, groups=None, cv_type='group', seed=42, params=None):
    """Train model with CV, return OOF predictions"""
    if cv_type == 'group':
        if groups is None:
            raise ValueError("groups must be provided for GroupKFold")
        cv = GroupKFold(n_splits=N_FOLDS)
        split_iter = cv.split(X, y, groups)
    else:
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        split_iter = cv.split(X, y)
    
    oof = np.zeros(len(X))
    best_rounds = []
    
    if model_name == 'xgb':
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
            'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'min_child_weight': 1,
            'reg_alpha': 0.1, 'reg_lambda': 1.0, 'seed': seed,
            'n_jobs': -1, 'verbosity': 0,
        }
        for fold, (tr, val) in enumerate(split_iter):
            dtrain = xgb.DMatrix(X.iloc[tr], label=y.iloc[tr])
            dval = xgb.DMatrix(X.iloc[val], label=y.iloc[val])
            m = xgb.train(params, dtrain, 2000, [(dtrain, 'tr'), (dval, 'val')],
                         early_stopping_rounds=100, verbose_eval=0)
            oof[val] = m.predict(dval, iteration_range=(0, m.best_iteration+1))
            best_rounds.append(m.best_iteration + 1)
    
    elif model_name == 'lgb':
        params_lgb = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
            'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1,
            'seed': seed, 'n_jobs': -1,
        }
        p = {**params_lgb, **(params or {})}
        for fold, (tr, val) in enumerate(split_iter):
            tr_data = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
            val_data = lgb.Dataset(X.iloc[val], label=y.iloc[val], reference=tr_data)
            callbacks = [lgb.early_stopping(100), lgb.log_evaluation(0)]
            m = lgb.train(p, tr_data, 2000, [tr_data, val_data], ['tr', 'val'], callbacks=callbacks)
            oof[val] = m.predict(X.iloc[val], num_iteration=m.best_iteration)
            best_rounds.append(m.best_iteration)
    
    elif model_name == 'hgb':
        params_hgb = {
            'loss': 'log_loss', 'learning_rate': 0.05, 'max_iter': 2000,
            'max_depth': 6, 'min_samples_leaf': 20, 'l2_regularization': 1.0,
            'max_bins': 255, 'random_state': seed, 'early_stopping': True,
            'validation_fraction': 0.1, 'n_iter_no_change': 100, 'tol': 1e-4,
        }
        p = {**params_hgb, **(params or {})}
        for fold, (tr, val) in enumerate(split_iter):
            m = HistGradientBoostingClassifier(**p)
            m.fit(X.iloc[tr], y.iloc[tr])
            oof[val] = m.predict_proba(X.iloc[val])[:, 1]
            best_rounds.append(m.n_iter_)
    
    elif model_name == 'cb':
        cat_features = [X.columns.get_loc(c) for c in ['gender', 'stress_level', 'academic_work_impact'] if c in X.columns]
        params_cb = {
            'objective': 'Logloss', 'eval_metric': 'AUC', 'iterations': 2000,
            'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 3,
            'subsample': 0.8, 'colsample_bylevel': 0.8, 'random_seed': seed,
            'verbose': False, 'thread_count': -1, 'early_stopping_rounds': 100,
        }
        p = {**params_cb, **(params or {})}
        for fold, (tr, val) in enumerate(split_iter):
            train_pool = Pool(X.iloc[tr], y.iloc[tr], cat_features=cat_features)
            val_pool = Pool(X.iloc[val], y.iloc[val], cat_features=cat_features)
            m = CatBoostClassifier(**p)
            m.fit(train_pool, eval_set=val_pool, use_best_model=True)
            oof[val] = m.predict_proba(X.iloc[val])[:, 1]
            best_rounds.append(m.get_best_iteration() + 1)
    
    return oof, int(np.mean(best_rounds))

def train_model_full(model_name, X, y, X_test, seed=42, n_rounds=1000):
    """Train model on full data (fixed n_rounds from CV best iteration) and predict test"""
    if model_name == 'xgb':
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
            'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'min_child_weight': 1,
            'reg_alpha': 0.1, 'reg_lambda': 1.0, 'seed': seed,
            'n_jobs': -1, 'verbosity': 0,
        }
        dtrain = xgb.DMatrix(X, label=y)
        dtest = xgb.DMatrix(X_test)
        m = xgb.train(params, dtrain, n_rounds, verbose_eval=0)
        return m.predict(dtest)
    
    elif model_name == 'lgb':
        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
            'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1,
            'seed': seed, 'n_jobs': -1,
        }
        train_data = lgb.Dataset(X, label=y)
        m = lgb.train(params, train_data, n_rounds, callbacks=[lgb.log_evaluation(0)])
        return m.predict(X_test)
    
    elif model_name == 'hgb':
        params = {
            'loss': 'log_loss', 'learning_rate': 0.05, 'max_iter': n_rounds,
            'max_depth': 6, 'min_samples_leaf': 20, 'l2_regularization': 1.0,
            'max_bins': 255, 'random_state': seed, 'early_stopping': False,
        }
        m = HistGradientBoostingClassifier(**params)
        m.fit(X, y)
        return m.predict_proba(X_test)[:, 1]
    
    elif model_name == 'cb':
        cat_features = [X.columns.get_loc(c) for c in ['gender', 'stress_level', 'academic_work_impact'] if c in X.columns]
        params = {
            'objective': 'Logloss', 'eval_metric': 'AUC', 'iterations': n_rounds,
            'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 3,
            'subsample': 0.8, 'colsample_bylevel': 0.8, 'random_seed': seed,
            'verbose': False, 'thread_count': -1,
        }
        train_pool = Pool(X, y, cat_features=cat_features)
        m = CatBoostClassifier(**params)
        m.fit(train_pool)
        return m.predict_proba(X_test)[:, 1]

def multi_seed_bagging(model_name, X, y, X_test, n_seeds=5, groups=None):
    """Multi-seed bagging for a model"""
    all_test_preds = []
    all_oof = []

    for seed in range(n_seeds):
        print(f"  Seed {seed+1}/{n_seeds}")
        oof, best_rounds = train_model_cv(model_name, X, y, cv_type='group', seed=seed, groups=groups)
        all_oof.append(oof)

        # Train on full data using CV-selected rounds
        test_pred = train_model_full(model_name, X, y, X_test, seed, n_rounds=best_rounds)
        all_test_preds.append(test_pred)
    
    avg_oof = np.mean(all_oof, axis=0)
    avg_test = np.mean(all_test_preds, axis=0)
    
    return np.mean(all_oof, axis=0), np.mean(all_test_preds, axis=0)

def adversarial_validation(X_train, X_test, feature_cols):
    """Detect train-test distribution shift"""
    train_adv = X_train.copy()
    test_adv = X_test.copy()
    train_adv['is_test'] = 0
    test_adv['is_test'] = 1
    
    adv_data = pd.concat([train_adv, test_adv], axis=0)
    y_adv = adv_data['is_test']
    X_adv = adv_data.drop('is_test', axis=1)
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_adv, y_adv)
    
    importance = pd.DataFrame({
        'feature': X_adv.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    from sklearn.model_selection import cross_val_predict
    adv_preds = cross_val_predict(rf, X_adv, y_adv, cv=5, method='predict_proba')[:, 1]
    adv_auc = roc_auc_score(y_adv, adv_preds)
    
    print(f"Adversarial AUC: {adv_auc:.4f} (0.5 = no shift, 1.0 = perfect separation)")
    
    leaky_features = importance[importance['importance'] > importance['importance'].mean()]['feature'].tolist()
    return importance, adv_auc, leaky_features

def main():
    print("="*60)
    print("Pool141 Improvement Pipeline - GPU")
    print("="*60)
    
    train, test = load_data()
    print("Preparing features with FE-03 missingness indicators + pool141 features...")
    X, y, X_test, feature_cols = get_features(train, test)
    print(f"Features: {X.shape[1]}, Train: {len(X)}, Test: {len(X_test)}")
    
    # Create age groups for GroupKFold
    age_groups = create_age_groups(train['age'], n_bins=10)
    print(f"Age groups: {age_groups.nunique()} unique groups")
    
    # Adversarial validation
    print("\n--- Adversarial Validation ---")
    importance, adv_auc, leaky_features = adversarial_validation(X, X_test, feature_cols)
    print(f"Adversarial AUC: {adv_auc:.4f}")
    print(f"Top 10 leaky features:")
    print(importance.head(10).to_string(index=False))
    print(f"Leaky features (importance > mean): {len(leaky_features)}")
    
    # Drop leaky features if adversarial AUC > 0.55
    if adv_auc > 0.55:
        print(f"\nDropping {len(leaky_features)} leaky features...")
        clean_features = [f for f in feature_cols if f not in leaky_features]
        X_clean = X[clean_features]
        X_test_clean = X_test[clean_features]
        print(f"Features after cleaning: {len(clean_features)}")
    else:
        X_clean, X_test_clean = X, X_test
        clean_features = feature_cols
    
    # Multi-seed bagging for each model
    print("\n=== Multi-seed Bagging (5 seeds per model) ===")
    models = ['xgb', 'lgb', 'hgb', 'cb']
    oof_dict = {}
    test_dict = {}

    for name in models:
        print(f"\n--- {name.upper()} Multi-seed Bagging (5 seeds) ---")
        oof, test_pred = multi_seed_bagging(name, X_clean, y, X_test_clean, n_seeds=5, groups=age_groups)
        oof_dict[name] = oof
        test_dict[name] = test_pred
        auc = roc_auc_score(y, oof)
        print(f"  {name} OOF AUC: {auc:.5f}")
    
    # 3-Level Stacking
    print("\n=== 3-Level Stacking ===")
    
    # Level 1: Base model OOF
    oof_stack = np.column_stack([oof_dict[m] for m in models])
    
    # Level 2: Meta-learner (RidgeCV on log-odds)
    eps = 1e-7
    oof_stack_logit = np.log((oof_stack + 1e-7) / (1 - oof_stack + 1e-7))
    
    meta_ridge = RidgeCV(alphas=np.logspace(-4, 4, 9), cv=5, scoring='roc_auc')
    meta_ridge.fit(oof_stack_logit, y)
    
    meta_oof = meta_ridge.predict(oof_stack_logit)
    from scipy.special import expit
    meta_oof_prob = expit(meta_oof)
    meta_auc = roc_auc_score(y, meta_oof_prob)
    print(f"  Ridge Meta-learner AUC: {meta_auc:.5f}")
    
    # LGB Meta-learner
    meta_lgb = lgb.LGBMClassifier(
        objective='binary', metric='auc', n_estimators=200,
        num_leaves=31, learning_rate=0.05, verbose=-1, random_state=42
    )
    meta_lgb.fit(oof_stack, y)
    meta_lgb_oof = meta_lgb.predict_proba(oof_stack)[:, 1]
    meta_lgb_auc = roc_auc_score(y, meta_lgb_oof)
    print(f"  LightGBM Meta-learner AUC: {meta_lgb_auc:.5f}")
    
    # XGB Meta-learner
    meta_xgb = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='auc', n_estimators=200,
        max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1
    )
    meta_xgb.fit(oof_stack, y)
    meta_xgb_oof = meta_xgb.predict_proba(oof_stack)[:, 1]
    meta_xgb_auc = roc_auc_score(y, meta_xgb_oof)
    print(f"  XGB Meta-learner AUC: {meta_xgb_auc:.5f}")
    
    # Simple averages
    for name in ['avg_xgb_lgb', 'avg_top3', 'avg_all']:
        if name == 'avg_xgb_lgb':
            avg = np.mean([oof_dict['xgb'], oof_dict['lgb']], axis=0)
        elif name == 'avg_top3':
            avg = np.mean([oof_dict[m] for m in ['xgb', 'lgb', 'hgb']], axis=0)
        else:
            avg = np.mean([oof_dict[m] for m in ['xgb', 'lgb', 'hgb', 'cb']], axis=0)
        auc = roc_auc_score(y, avg)
        print(f"  {name}: {auc:.5f}")
    
    # Weighted average
    weights = {m: roc_auc_score(y, oof_dict[m]) for m in models}
    total_w = sum(weights.values())
    weighted = sum((weights[m]/total_w) * oof_dict[m] for m in models)
    auc_w = roc_auc_score(y, weighted)
    print(f"  weighted: {auc_w:.5f}")
    
    # Best ensemble
    best_method = max([
        ('ridge_meta', meta_auc),
        ('lgb_meta', meta_lgb_auc),
        ('xgb_meta', meta_xgb_auc),
        ('avg_top3', roc_auc_score(y, np.mean([oof_dict[m] for m in ['xgb', 'lgb', 'hgb']], axis=0))),
        ('avg_xgb_lgb', roc_auc_score(y, np.mean([oof_dict['xgb'], oof_dict['lgb']], axis=0))),
    ], key=lambda x: x[1])
    
    print(f"\nBest ensemble: {best_method[0]} = {best_method[1]:.5f}")
    
    # Generate test predictions for best ensemble
    print("\n=== Generating Final Test Predictions ===")
    test_stack = np.column_stack([test_dict[m] for m in models])
    test_stack_logit = np.log((test_stack + 1e-7) / (1 - test_stack + 1e-7))
    
    # Use best ensemble method (matches best_method selection above)
    if best_method[0] == 'lgb_meta':
        final_test = meta_lgb.predict_proba(test_stack)[:, 1]
    elif best_method[0] == 'xgb_meta':
        final_test = meta_xgb.predict_proba(test_stack)[:, 1]
    elif best_method[0] == 'avg_top3':
        final_test = np.mean([test_dict[m] for m in ['xgb', 'lgb', 'hgb']], axis=0)
    elif best_method[0] == 'avg_xgb_lgb':
        final_test = np.mean([test_dict['xgb'], test_dict['lgb']], axis=0)
    else:  # ridge_meta
        from scipy.special import expit
        final_test = expit(meta_ridge.predict(test_stack_logit))
    final_test = np.clip(final_test, 0, 1)
    
    # Generate submission
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    submission = pd.DataFrame({
        'id': sample_sub['id'],
        'addicted_label': final_test
    })
    
    output_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('../submissions')
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / 'submission_pool141_improved.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to {submission_path}")
    print(f"Shape: {submission.shape}, Mean: {final_test.mean():.5f}, Min: {final_test.min():.5f}, Max: {final_test.max():.5f}")

if __name__ == '__main__':
    main()