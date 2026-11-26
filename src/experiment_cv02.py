"""
CV Strategy Experiment: CV-02
Adversarial validation: detect train-test distribution shift, drop leaky features
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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
    return X, y, feature_cols

def adversarial_validation(X_train, X_test, feature_cols):
    """Train classifier to distinguish train vs test, get feature importance"""
    train_adv = X_train[feature_cols].copy()
    test_adv = X_test[feature_cols].copy()
    
    train_adv['is_test'] = 0
    test_adv['is_test'] = 1
    
    adv_data = pd.concat([train_adv, test_adv], axis=0)
    y_adv = adv_data['is_test']
    X_adv = adv_data.drop('is_test', axis=1)
    
    # Train RF to distinguish train vs test
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_adv, y_adv)
    
    # Get feature importances
    importance = pd.DataFrame({
        'feature': X_adv.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Adversarial AUC
    from sklearn.model_selection import cross_val_predict
    adv_preds = cross_val_predict(rf, X_adv, y_adv, cv=5, method='predict_proba')[:, 1]
    adv_auc = roc_auc_score(y_adv, adv_preds)
    
    print(f"Adversarial AUC: {adv_auc:.4f} (0.5 = no shift, 1.0 = perfect separation)")
    
    # Features with high importance are leaky/different between train/test
    leaky_features = importance[importance['importance'] > importance['importance'].mean()]['feature'].tolist()
    
    return importance, adv_auc, leaky_features

def train_xgb_cv(X, y, feature_names=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
        'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'min_child_weight': 1,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42,
        'n_jobs': -1, 'verbosity': 0,
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        model = xgb.train(params, dtrain, num_boost_round=2000,
                        evals=[(dtrain, 'train'), (dval, 'valid')],
                        early_stopping_rounds=100, verbose_eval=False)
        oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    
    return oof_preds

def main():
    print("="*60)
    print("CV-02: Adversarial Validation")
    print("="*60)
    
    train, test = load_data()
    print("Preparing features with FE-03 missingness indicators...")
    X, y, feature_cols = get_features(train, test)
    # Process test data for adversarial validation
    num_cols = test.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['id', 'addicted_label']]
    test_proc = add_missingness_indicators(test, num_cols)
    test_processed = preprocess_basic(test_proc)
    X_test_processed = test_processed[feature_cols]
    
    print(f"Features: {len(feature_cols)}, Train: {len(X)}, Test: {len(X_test_processed)}")
    
    # 1. Adversarial validation on processed features
    print("\n--- Adversarial Validation (Processed Features) ---")
    importance, adv_auc, leaky_features = adversarial_validation(X, X_test_processed, feature_cols)
    print(f"Adversarial AUC: {adv_auc:.4f}")
    print(f"Top 10 leaky features:")
    print(importance.head(10).to_string(index=False))
    print(f"\nLeaky features (importance > mean): {len(leaky_features)}")
    for f in leaky_features[:15]:
        print(f"  {f}")
    
    # 2. Train baseline with all features
    print("\n--- Baseline XGBoost (All Features) ---")
    oof_all = train_xgb_cv(X, y, feature_cols)
    baseline_auc = roc_auc_score(y, oof_all)
    print(f"Baseline AUC: {baseline_auc:.5f}")
    
    # 3. Drop leaky features and retrain
    if leaky_features:
        print(f"\n--- XGBoost without leaky features ({len(leaky_features)} dropped) ---")
        clean_features = [f for f in feature_cols if f not in leaky_features]
        X_clean = X[clean_features]
        oof_clean = train_xgb_cv(X_clean, y, clean_features)
        clean_auc = roc_auc_score(y, oof_clean)
        print(f"Clean features AUC: {clean_auc:.5f}")
        print(f"Change: {clean_auc - baseline_auc:+.5f}")
    
    # 4. Try dropping top N most leaky features progressively
    print("\n--- Progressive feature removal ---")
    importance_sorted = importance.sort_values('importance', ascending=False)
    
    results = []
    results.append(('baseline', len(feature_cols), baseline_auc))
    
    for n_drop in [5, 10, 15, 20]:
        drop_features = importance_sorted.head(n_drop)['feature'].tolist()
        clean_features = [f for f in feature_cols if f not in drop_features]
        if len(clean_features) < 5:
            break
        X_clean = X[clean_features]
        oof_clean = train_xgb_cv(X_clean, y, clean_features)
        clean_auc = roc_auc_score(y, oof_clean)
        print(f"Drop top {n_drop}: {clean_auc:.5f} (features: {len(clean_features)})")
        results.append((f'drop_top_{n_drop}', len(clean_features), clean_auc))
    
    # Save results
    results_df = pd.DataFrame(results, columns=['config', 'n_features', 'cv_auc'])
    results_df['exp_id'] = 'CV-02'
    results_df['category'] = 'CV Strategy'
    results_df['description'] = 'Adversarial validation feature removal'
    results_df['adv_auc'] = adv_auc
    results_df['status'] = 'completed'
    
    logs_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('../logs')
    logs_dir.mkdir(exist_ok=True)
    exp_file = logs_dir / 'experiments.csv'
    if exp_file.exists():
        existing = pd.read_csv(exp_file)
        results_df = pd.concat([existing, results_df], ignore_index=True)
    results_df.to_csv(exp_file, index=False)
    print(f"\nResults saved to {exp_file}")
    
    # Save feature importance for reference
    importance_path = logs_dir / 'cv02_feature_importance.csv'
    importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved to {importance_path}")

if __name__ == '__main__':
    main()