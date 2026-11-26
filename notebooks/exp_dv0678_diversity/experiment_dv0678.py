"""
DV-06/07/08: Own-track family diversity models with SAVED OOF + test preds.
Members: ExtraTrees, RandomForest, LightGBM-dart, EBM (if installable).
Features + CV identical to pool141_improve (FE-03 + pool141 feats, GroupKFold-age).
Each member: 3-seed bagged, saves oof_<name>.csv and test_<name>.csv with id column
so later hill-climbing / pool injection has matched pairs.
"""
import os, subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
import lightgbm as lgb

if os.path.exists('/kaggle/input/competitions/playground-series-s6e8/train.csv'):
    DATA_DIR = Path('/kaggle/input/competitions/playground-series-s6e8')
elif os.path.exists('../data/train.csv'):
    DATA_DIR = Path('../data')
else:
    DATA_DIR = Path('data')

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    HAS_EBM = True
except ImportError:
    try:
        subprocess.run(['pip', 'install', '-q', 'interpret'], check=True)
        from interpret.glassbox import ExplainableBoostingClassifier
        HAS_EBM = True
    except Exception as e:
        print(f"EBM unavailable: {e}")
        HAS_EBM = False

SEED = 42
N_FOLDS = 5
N_SEEDS = 3

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
    return train[feature_cols], train['addicted_label'], test[feature_cols]

def create_age_groups(age_series, n_bins=10):
    age_filled = age_series.fillna(age_series.median())
    groups = pd.qcut(age_filled, q=n_bins, labels=False, duplicates='drop')
    if groups.isna().any():
        groups = pd.cut(age_filled, bins=n_bins, labels=False)
    return groups.astype(int)

def make_model(name, seed):
    if name == 'et':
        return ExtraTreesClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20,
                                    max_features='sqrt', random_state=seed, n_jobs=-1)
    if name == 'rf':
        return RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20,
                                      max_features=0.6, random_state=seed, n_jobs=-1)
    if name == 'dart':
        return lgb.LGBMClassifier(objective='binary', boosting_type='dart', num_leaves=63,
                                  learning_rate=0.05, n_estimators=600, feature_fraction=0.8,
                                  bagging_fraction=0.8, bagging_freq=5, verbose=-1,
                                  random_state=seed, n_jobs=-1)
    if name == 'ebm':
        return ExplainableBoostingClassifier(random_state=seed, n_jobs=-1)
    raise ValueError(name)

def run_member(name, X, y, X_test, groups):
    oof_acc, test_acc = [], np.zeros(len(X_test))
    for seed in range(SEED, SEED + N_SEEDS):
        print(f"  {name} seed {seed}")
        oof = np.zeros(len(X))
        cv = GroupKFold(n_splits=N_FOLDS)
        for tr, val in cv.split(X, y, groups):
            m = make_model(name, seed)
            m.fit(X.iloc[tr], y.iloc[tr])
            oof[val] = m.predict_proba(X.iloc[val])[:, 1]
        oof_acc.append(oof)
        m = make_model(name, seed)
        m.fit(X, y)
        test_acc += m.predict_proba(X_test)[:, 1] / N_SEEDS
    oof = np.mean(oof_acc, axis=0)
    print(f"  {name} OOF AUC: {roc_auc_score(y, oof):.5f}")
    return oof, test_acc

def main():
    print("=" * 60)
    print("DV-06/07/08: Diversity members (ET, RF, dart, EBM)")
    print("=" * 60)
    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    X, y, X_test = get_features(train, test)
    groups = create_age_groups(train['age'])
    print(f"Features: {X.shape[1]}, Train: {len(X)}, Test: {len(X_test)}")

    out = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('../models')
    out.mkdir(parents=True, exist_ok=True)

    names = ['et', 'rf', 'dart'] + (['ebm'] if HAS_EBM else [])
    for name in names:
        print(f"\n--- {name.upper()} ---")
        oof, test_pred = run_member(name, X, y, X_test, groups)
        pd.DataFrame({'id': train['id'], f'{name}_oof': oof}).to_csv(out / f'oof_{name}.csv', index=False)
        pd.DataFrame({'id': test['id'], f'{name}_test': test_pred}).to_csv(out / f'test_{name}.csv', index=False)
        print(f"  saved oof_{name}.csv / test_{name}.csv")

    print("\nDone.")

if __name__ == '__main__':
    main()
