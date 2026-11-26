"""
Model Diversity Experiment: MD-05
Random Forest / ExtraTrees with FE-03 features
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Kaggle kernel: competition data is at /kaggle/input/competitions/playground-series-s6e8/
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

def train_rf_cv(X, y, model_type='rf'):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    fold_scores = []
    
    if model_type == 'rf':
        params = {
            'n_estimators': 500,
            'max_depth': 12,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': SEED,
            'n_jobs': -1,
            'class_weight': 'balanced_subsample',
        }
        model_class = RandomForestClassifier
    else:
        params = {
            'n_estimators': 500,
            'max_depth': 12,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': SEED,
            'n_jobs': -1,
            'class_weight': 'balanced_subsample',
        }
        model_class = ExtraTreesClassifier
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = model_class(**params)
        model.fit(X_train, y_train)
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_scores.append(fold_auc)
        print(f"  Fold {fold+1}: AUC = {fold_auc:.5f}")
    
    cv_auc = roc_auc_score(y, oof_preds)
    print(f"  CV AUC: {cv_auc:.5f} (+/- {np.std(fold_scores):.5f})")
    return cv_auc, oof_preds

def main():
    print("="*60)
    print("MD-05: Random Forest & ExtraTrees (FE-03 Features)")
    print("="*60)
    
    train, test = load_data()
    
    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['id', 'addicted_label']]
    
    print("Adding FE-03 missingness indicators...")
    train = add_missingness_indicators(train, num_cols)
    test = add_missingness_indicators(test, num_cols)
    
    train = preprocess_basic(train)
    test = preprocess_basic(test)
    
    feature_cols = [c for c in train.columns if c not in ['id', 'addicted_label']]
    X = train[feature_cols]
    y = train['addicted_label']
    
    missing_indicator_cols = [c for c in feature_cols if c.endswith('_was_missing')]
    print(f"Features: {len(feature_cols)} (added {len(missing_indicator_cols)} missingness indicators)")
    print(f"Train samples: {len(X)}")
    
    print("\nTraining Random Forest with 5-fold CV...")
    rf_auc, rf_oof = train_rf_cv(X, y, 'rf')
    
    print("\nTraining ExtraTrees with 5-fold CV...")
    et_auc, et_oof = train_rf_cv(X, y, 'et')
    
    # Save both
    for model_name, cv_auc, oof_preds in [('rf', rf_auc, rf_oof), ('et', et_auc, et_oof)]:
        results = pd.DataFrame({
            'exp_id': [f'MD-05-{model_name}'],
            'category': ['Model Div'],
            'description': [f'{model_name.upper()} with FE-03 missingness indicators'],
            'cv_auc': [cv_auc],
            'n_features': [len(feature_cols)],
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
        
        oof_df = pd.DataFrame({
            'id': train['id'],
            f'md05_{model_name}_oof': oof_preds
        })
        oof_path = Path(f'/kaggle/working/md05_{model_name}_oof.csv') if Path('/kaggle/working').exists() else Path(f'../models/md05_{model_name}_oof.csv')
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        oof_df.to_csv(oof_path, index=False)
        print(f"OOF predictions saved to {oof_path}")

if __name__ == '__main__':
    main()