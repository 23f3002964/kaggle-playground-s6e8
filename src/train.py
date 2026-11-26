"""
Training script for Playground Series S6E8 - Predicting Smartphone Addiction
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# Config
DATA_DIR = Path('../data')
MODEL_DIR = Path('../models')
SUBMISSION_DIR = Path('../submissions')
MODEL_DIR.mkdir(exist_ok=True)
SUBMISSION_DIR.mkdir(exist_ok=True)

SEED = 42
N_FOLDS = 5


def load_data():
    """Load train and test data"""
    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    return train, test


def preprocess_data(train, test):
    """Basic preprocessing: handle missing values, encode categoricals"""
    # Combine for consistent preprocessing
    train['is_train'] = 1
    test['is_train'] = 0
    test['addicted_label'] = -1
    
    df = pd.concat([train, test], axis=0, ignore_index=True)
    
    # Handle missing values in numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['id', 'addicted_label', 'is_train']]
    
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Handle missing values in categorical columns
    cat_cols = ['gender', 'stress_level', 'academic_work_impact']
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Split back
    train_processed = df[df['is_train'] == 1].drop(['is_train'], axis=1).reset_index(drop=True)
    test_processed = df[df['is_train'] == 0].drop(['is_train', 'addicted_label'], axis=1).reset_index(drop=True)
    
    return train_processed, test_processed


def train_model(X_train, y_train, X_val, y_val, params=None):
    """Train LightGBM model"""
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': SEED,
            'n_jobs': -1,
        }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ]
    )
    
    return model


def main():
    print("Loading data...")
    train, test = load_data()
    
    print("Preprocessing...")
    train_proc, test_proc = preprocess_data(train, test)
    
    # Prepare features and target
    feature_cols = [c for c in train_proc.columns if c not in ['id', 'addicted_label']]
    X = train_proc[feature_cols]
    y = train_proc['addicted_label']
    X_test = test_proc[feature_cols]
    
    print(f"Features: {len(feature_cols)}")
    print(f"Train samples: {len(X)}")
    print(f"Test samples: {len(X_test)}")
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{N_FOLDS}")
        print(f"{'='*50}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = train_model(X_train, y_train, X_val, y_val)
        
        # OOF predictions
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_scores.append(fold_auc)
        print(f"Fold {fold + 1} AUC: {fold_auc:.5f}")
        
        # Test predictions
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
        
        # Save model
        model.save_model(MODEL_DIR / f'lgb_fold_{fold}.txt')
    
    print(f"\n{'='*50}")
    print(f"CV AUC: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")
    print(f"OOF AUC: {roc_auc_score(y, oof_preds):.5f}")
    print(f"{'='*50}")
    
    # Generate submission
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    submission = pd.DataFrame({
        'id': sample_sub['id'],
        'addicted_label': test_preds
    })
    
    submission_path = SUBMISSION_DIR / f'submission_lgb_cv_{np.mean(fold_scores):.5f}.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")
    
    # Save OOF predictions for stacking
    oof_df = pd.DataFrame({
        'id': train_proc['id'],
        'addicted_label': oof_preds
    })
    oof_df.to_csv(MODEL_DIR / 'oof_preds.csv', index=False)
    

if __name__ == '__main__':
    main()