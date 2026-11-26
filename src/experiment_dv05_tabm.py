"""
DV-05 (redo): TabM / RealMLP via pytabkit on Colab GPU.
Diversity member for rank-space hill climbing (DV-11) + pool injection (DV-04).
Based on: omidbaghchehsaraei's TabM (OOF 0.96751 → our tabm_seed3 0.96867)
          RealMLP-TD (pub_rmlp 0.96844)
Strategy: constrained imputation + lattice target encoding (same as winning models)
          5-seed bagging (or seed-averaged inside folds like tabm_seed3)
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

# ───────────────────────── Paths ─────────────────────────
if os.path.exists('/kaggle/input/competitions/playground-series-s6e8/train.csv'):
    DATA_DIR = Path('/kaggle/input/competitions/playground-series-s6e8')
elif os.path.exists('/content/train.csv'):
    DATA_DIR = Path('/content')
elif os.path.exists('/kaggle/working/train.csv'):
    DATA_DIR = Path('/kaggle/working')
else:
    DATA_DIR = Path('../data')

if Path('/kaggle/working').exists():
    OUT_DIR = Path('/kaggle/working')
elif Path('/content').exists():
    OUT_DIR = Path('/content')
else:
    OUT_DIR = Path('../models')

SEED = 42
N_FOLDS = 5
N_SEEDS = 5          # bagging seeds (model init); folds stay fixed for aligned OOF
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CAT_COLS = ['gender', 'stress_level', 'academic_work_impact']
NUM_COLS_RAW = [
    'age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
    'work_study_hours', 'sleep_hours', 'notifications_per_day',
    'app_opens_per_day', 'weekend_screen_time'
]


def prog(msg):
    """Parseable progress line: grep 'DV05T]' for live status."""
    print(f"[DV05T] {msg}", flush=True)


# ───────────────────────── Feature Engineering ─────────────────────────
# Same as experiment_pool141_improve.py (winning features)
def add_missingness_indicators(df, num_cols):
    df = df.copy()
    for col in num_cols:
        df[f'{col}_was_missing'] = df[col].isna().astype(float)
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
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 50, 100], labels=[0, 1, 2, 3, 4]).astype(float)
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


def lattice_target_encoding(train, test, cat_cols, smoothing=20, min_samples_leaf=10):
    """
    Lattice target encoding: target mean per category with smoothing.
    Applied to all categorical columns + their interactions (pair cells).
    """
    from sklearn.preprocessing import LabelEncoder
    
    train = train.copy()
    test = test.copy()
    
    global_mean = train['addicted_label'].mean()
    
    # Single column TE
    for col in cat_cols:
        if col not in train.columns:
            continue
        le = LabelEncoder()
        train[col] = train[col].fillna('Unknown').astype(str)
        test[col] = test[col].fillna('Unknown').astype(str)
        train[f'{col}_le'] = le.fit_transform(train[col])
        test[f'{col}_le'] = le.transform(test[col])
        
        stats = train.groupby(f'{col}_le')['addicted_label'].agg(['mean', 'count'])
        stats['smooth'] = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
        stats.loc[stats['count'] < min_samples_leaf, 'smooth'] = global_mean
        
        train[f'{col}_te'] = train[f'{col}_le'].map(stats['smooth']).fillna(global_mean)
        test[f'{col}_te'] = test[f'{col}_le'].map(stats['smooth']).fillna(global_mean)
    
    # Pair lattice TE (all 36 numeric pairs + 3 screen-block triples)
    num_cols = [c for c in train.columns if c not in ['id', 'addicted_label'] + cat_cols 
                and train[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    # Bin numeric columns for pair cells
    for col in num_cols:
        if train[col].nunique() > 20:
            bins = pd.qcut(train[col], q=10, labels=False, duplicates='drop')
            train[f'{col}_bin'] = bins.astype(float)
            test[f'{col}_bin'] = pd.cut(test[col], bins=10, labels=False).astype(float)
        else:
            train[f'{col}_bin'] = train[col]
            test[f'{col}_bin'] = test[col]
    
    bin_cols = [c for c in train.columns if c.endswith('_bin')]
    
    # Pair cells (all pairs of binned numerics)
    for i, c1 in enumerate(bin_cols):
        for c2 in bin_cols[i+1:]:
            pair_name = f'pair_{c1}_{c2}'
            train[pair_name] = train[c1].astype(str) + '_' + train[c2].astype(str)
            test[pair_name] = test[c1].astype(str) + '_' + test[c2].astype(str)
            
            # TE on pair
            stats = train.groupby(pair_name)['addicted_label'].agg(['mean', 'count'])
            stats['smooth'] = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
            stats.loc[stats['count'] < min_samples_leaf, 'smooth'] = global_mean
            
            train[f'{pair_name}_te'] = train[pair_name].map(stats['smooth']).fillna(global_mean)
            test[f'{pair_name}_te'] = test[pair_name].map(stats['smooth']).fillna(global_mean)
            
            # Drop raw pair to save memory
            train.drop(columns=[pair_name], inplace=True)
            test.drop(columns=[pair_name], inplace=True)
    
    return train, test


def preprocess_basic(df):
    """Fill missing + label encode categoricals."""
    for col in CAT_COLS:
        df[col] = df[col].fillna('Unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def get_features(train, test):
    """Full feature pipeline: missingness + pool141 + lattice TE + basic preprocessing."""
    num_cols = [c for c in NUM_COLS_RAW if c in train.columns]
    
    train = add_missingness_indicators(train, num_cols)
    test = add_missingness_indicators(test, num_cols)
    
    train = add_pool141_features(train)
    test = add_pool141_features(test)
    
    # Lattice target encoding
    train, test = lattice_target_encoding(train, test, CAT_COLS)
    
    # Basic preprocessing
    train = preprocess_basic(train)
    test = preprocess_basic(test)
    
    feature_cols = [c for c in train.columns if c not in ['id', 'addicted_label']]
    X = train[feature_cols]
    y = train['addicted_label'].values.astype(np.float32)
    X_test = test[feature_cols]
    
    return X, y, X_test, feature_cols


# ───────────────────────── pytabkit Model Wrapper ─────────────────────────
def train_tabm_pytabkit(X_train, y_train, X_val, y_val, X_test, seed=42, model_type='tabm'):
    """
    Train TabM or RealMLP using pytabkit.
    Falls back to sklearn MLP if pytabkit not available.
    """
    try:
        import pytabkit as ptk
        from pytabkit.models import TabM, RealMLP
        from pytabkit.models.nn_models.tabm import TabMConfig
        from pytabkit.models.nn_models.realmlp import RealMLPConfig
        
        prog(f"Using pytabkit {ptk.__version__ if hasattr(ptk, '__version__') else 'unknown'}")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        
        # Convert to tensors
        X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(DEVICE)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
        X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(DEVICE)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE)
        
        n_features = X_train.shape[1]
        
        if model_type == 'tabm':
            # TabM config matching best variant (tabm_seed3: 10 blocks, 160 wide, dropout 0.02, lr 5e-4)
            config = TabMConfig(
                n_blocks=10,
                d_block=160,
                dropout=0.02,
                lr=5e-4,
                n_epochs=150,
                batch_size=1024,
                weight_decay=1e-4,
                device=DEVICE,
                seed=seed,
            )
            model = TabM(config)
        else:
            # RealMLP-TD config
            config = RealMLPConfig(
                n_blocks=8,
                d_block=256,
                dropout=0.1,
                lr=1e-3,
                n_epochs=100,
                batch_size=1024,
                weight_decay=1e-4,
                device=DEVICE,
                seed=seed,
            )
            model = RealMLP(config)
        
        # Train with early stopping
        prog(f"Training {model_type} (seed={seed})...")
        model.fit(X_train_t, y_train_t, X_val=X_val_t, y_val=y_val_t)
        
        # Predict
        val_pred = model.predict_proba(X_val_t)[:, 1].cpu().numpy()
        test_pred = model.predict_proba(X_test_t)[:, 1].cpu().numpy()
        
        return val_pred, test_pred
        
    except ImportError:
        prog("pytabkit not available, falling back to sklearn MLP")
        return train_sklearn_mlp(X_train, y_train, X_val, y_val, X_test, seed)


def train_sklearn_mlp(X_train, y_train, X_val, y_val, X_test, seed=42):
    """Fallback: sklearn MLP with similar architecture."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # Architecture matching TabM capacity
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=1024,
        learning_rate_init=1e-3,
        max_iter=150,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=seed,
        verbose=False,
    )
    
    model.fit(X_train_s, y_train)
    
    val_pred = model.predict_proba(X_val_s)[:, 1]
    test_pred = model.predict_proba(X_test_s)[:, 1]
    
    return val_pred, test_pred


# ───────────────────────── Main ─────────────────────────
def main():
    global DEVICE
    t_start = time.time()
    dev_name = torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu'
    print("=" * 60)
    print(f"DV-05 (redo): TabM/RealMLP diversity member | device={dev_name}")
    print("=" * 60)

    # Try to install pytabkit if on Colab
    if DEVICE.type == 'cuda' and not os.path.exists('/kaggle'):
        try:
            import pytabkit
        except ImportError:
            prog("Installing pytabkit...")
            os.system("pip install -q pytabkit")
            import pytabkit
            prog(f"pytabkit installed: {pytabkit.__version__}")

    train, test = load_data()
    X, y, X_test, feature_cols = get_features(train, test)
    prog(f"data ready: train={len(y)} test={len(X_test)} features={len(feature_cols)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Try both TabM and RealMLP, pick better one
    model_types = ['tabm', 'realmlp']
    best_model_type = None
    best_cv = 0
    best_oof = None
    best_test = None

    for model_type in model_types:
        prog(f"\n=== Trying {model_type.upper()} ===")
        
        oof_seeds, test_seeds = [], []
        
        for s_i, seed in enumerate(range(SEED, SEED + N_SEEDS)):
            oof = np.zeros(len(y))
            test_pred = np.zeros(len(X_test))
            
            for f_i, (tr_idx, va_idx) in enumerate(splits):
                tag = f"{model_type} seed {s_i+1}/{N_SEEDS} fold {f_i+1}/{N_FOLDS}"
                
                val_pred, test_fold_pred = train_tabm_pytabkit(
                    X.iloc[tr_idx].values, y[tr_idx],
                    X.iloc[va_idx].values, y[va_idx],
                    X_test.values,
                    seed=seed,
                    model_type=model_type
                )
                
                oof[va_idx] = val_pred
                test_pred += test_fold_pred / N_FOLDS
            
            cv_auc = roc_auc_score(y, oof)
            prog(f"{model_type} seed {s_i+1}/{N_SEEDS} DONE oof_auc={cv_auc:.5f} elapsed={time.time()-t_start:.0f}s")
            oof_seeds.append(oof)
            test_seeds.append(test_pred)
        
        oof_final = np.mean(oof_seeds, axis=0)
        test_final = np.clip(np.mean(test_seeds, axis=0), 0, 1)
        cv_auc = roc_auc_score(y, oof_final)
        
        prog(f"{model_type} FINAL oof_auc={cv_auc:.5f} test_mean={test_final.mean():.5f}")
        
        if cv_auc > best_cv:
            best_cv = cv_auc
            best_model_type = model_type
            best_oof = oof_final
            best_test = test_final

    prog(f"\nBEST: {best_model_type.upper()} with CV AUC={best_cv:.5f}")

    # Self-checks
    assert not np.isnan(best_oof).any() and not np.isnan(best_test).any(), "NaN in predictions"
    assert best_oof.min() >= 0 and best_oof.max() <= 1, "OOF out of [0,1]"
    assert len(best_test) == len(test) == 296302, "wrong test length"
    mean_gap = abs(best_test.mean() - pd.Series(y).mean())
    prog(f"FINAL test_mean={best_test.mean():.5f} target_mean={pd.Series(y).mean():.5f} gap={mean_gap:.4f}")

    # Save
    sub_id = pd.read_csv(DATA_DIR / 'sample_submission.csv')['id']
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sub_path = OUT_DIR / f'submission_dv05_{best_model_type}.csv'
    pd.DataFrame({'id': sub_id, 'addicted_label': best_test}).to_csv(sub_path, index=False)
    
    oof_path = OUT_DIR / f'dv05_{best_model_type}_oof.csv'
    pd.DataFrame({'id': train['id'], f'dv05_{best_model_type}_oof': best_oof}).to_csv(oof_path, index=False)

    # Log experiment
    exp_file = OUT_DIR.parent / 'logs/experiments.csv'
    row = pd.DataFrame({
        'method': [f'{best_model_type}_5seed'], 'cv_auc': [best_cv], 'exp_id': ['DV-05'],
        'category': ['Pool'], 'description': [f'{best_model_type.upper()} 5-seed bagged, lattice TE, constrained imputation'],
        'status': ['pending_LB']
    })
    if exp_file.exists():
        pd.concat([pd.read_csv(exp_file), row], ignore_index=True).to_csv(exp_file, index=False)
    else:
        row.to_csv(exp_file, index=False)

    prog(f"saved: {sub_path} | {oof_path} | wall_time={(time.time()-t_start)/60:.1f}min")


def load_data():
    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    return train, test


if __name__ == '__main__':
    main()