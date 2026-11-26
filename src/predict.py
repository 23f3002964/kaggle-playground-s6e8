"""
Prediction script for generating submissions from trained models
"""
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb

MODEL_DIR = Path('../models')
SUBMISSION_DIR = Path('../submissions')
DATA_DIR = Path('../data')
SUBMISSION_DIR.mkdir(exist_ok=True)


def load_test_data():
    """Load and preprocess test data"""
    test = pd.read_csv(DATA_DIR / 'test.csv')
    
    # Same preprocessing as train
    num_cols = test.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != 'id']
    
    for col in num_cols:
        test[col] = test[col].fillna(test[col].median())
    
    cat_cols = ['gender', 'stress_level', 'academic_work_impact']
    from sklearn.preprocessing import LabelEncoder
    for col in cat_cols:
        test[col] = test[col].fillna('Unknown')
        le = LabelEncoder()
        test[col] = le.fit_transform(test[col].astype(str))
    
    feature_cols = [c for c in test.columns if c != 'id']
    return test[feature_cols], test['id']


def predict():
    """Generate predictions using saved models"""
    X_test, test_ids = load_test_data()
    
    # Find all model files
    model_files = list(MODEL_DIR.glob('lgb_fold_*.txt'))
    if not model_files:
        raise FileNotFoundError("No model files found in ../models/")
    
    print(f"Found {len(model_files)} models")
    
    # Average predictions
    test_preds = np.zeros(len(X_test))
    
    for model_file in model_files:
        print(f"Loading {model_file.name}...")
        model = lgb.Booster(model_file=str(model_file))
        preds = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds += preds / len(model_files)
    
    # Generate submission
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    submission = pd.DataFrame({
        'id': sample_sub['id'],
        'addicted_label': test_preds
    })
    
    submission_path = SUBMISSION_DIR / 'submission_ensemble.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")
    
    return submission


if __name__ == '__main__':
    predict()