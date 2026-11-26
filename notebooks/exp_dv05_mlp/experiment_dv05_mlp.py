"""
DV-05: PyTorch MLP (3-layer: 256-128-64) with categorical embeddings.
Diversity member for rank-space hill climbing (DV-11) + pool injection (DV-04).
Spec: DIVERSITY_PLAN.md Phase 2 — standardized numerics, cat embeddings,
5-seed bagged, Colab GPU. Progress lines are parseable for status checks.
"""
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

if os.path.exists('/kaggle/input/competitions/playground-series-s6e8/train.csv'):
    DATA_DIR = Path('/kaggle/input/competitions/playground-series-s6e8')
elif os.path.exists('/kaggle/input/playground-series-s6e8/train.csv'):
    DATA_DIR = Path('/kaggle/input/playground-series-s6e8')
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
N_SEEDS = 5          # bagging seeds (model init/dropout); folds stay fixed for aligned OOF
MAX_EPOCHS = 15
PATIENCE = 3
BATCH_SIZE = 1024
LR = 1e-3
EMB_DIM = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CAT_COLS = ['gender', 'stress_level', 'academic_work_impact']


def prog(msg):
    """Parseable progress line: grep 'DV05]' for live status."""
    print(f"[DV05] {msg}", flush=True)


def load_data():
    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    return train, test


def get_features(train, test):
    """Missingness indicators -> median/'Unknown' fill -> shared cat vocab (train-fitted)."""
    num_cols = [c for c in train.select_dtypes(include=[np.number]).columns
                if c not in ['id', 'addicted_label']]
    for df in (train, test):
        for col in num_cols:
            df[f'{col}_was_missing'] = df[col].isna().astype(float)
            df[col] = df[col].fillna(train[col].median())
        for col in CAT_COLS:
            df[col] = df[col].fillna('Unknown').astype(str)

    # Leakage-safe vocab: fitted on train only, unseen test values -> index 0
    cat_cards, cat_maps = [], []
    for col in CAT_COLS:
        vals = sorted(train[col].unique())
        m = {v: i + 1 for i, v in enumerate(vals)}
        for df in (train, test):
            df[col] = df[col].map(m).fillna(0).astype(int)
        cat_cards.append(len(vals) + 1)
        cat_maps.append(m)

    feature_cols = [c for c in train.columns if c not in ['id', 'addicted_label']]
    num_features = [c for c in feature_cols if c not in CAT_COLS]
    return (train[num_features], train[CAT_COLS], train['addicted_label'].values.astype(np.float32),
            test[num_features], test[CAT_COLS], num_features, cat_cards)


def to_tensors(Xn, Xc):
    return (torch.tensor(np.asarray(Xn), dtype=torch.float32),
            torch.tensor(np.asarray(Xc), dtype=torch.long))


class MLP(nn.Module):
    """Numerics branch (BN->256) + cat embeddings -> 128 -> 64 -> 1 logit."""

    def __init__(self, n_num, cat_cards):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, EMB_DIM) for c in cat_cards])
        self.bn = nn.BatchNorm1d(n_num)
        emb_out = EMB_DIM * len(cat_cards)
        self.net = nn.Sequential(
            nn.Linear(n_num + emb_out, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, xn, xc):
        e = torch.cat([emb(xc[:, i]) for i, emb in enumerate(self.embs)], dim=1)
        return self.net(torch.cat([self.bn(xn), e], dim=1)).squeeze(1)


@torch.no_grad()
def predict(model, Xn, Xc):
    model.eval()
    outs = []
    for i in range(0, len(Xn), 8192):
        outs.append(torch.sigmoid(model(Xn[i:i + 8192].to(DEVICE),
                                        Xc[i:i + 8192].to(DEVICE))).cpu().numpy())
    return np.concatenate(outs)


def train_fold(seed, Xn_tr, Xc_tr, y_tr, Xn_va, Xc_va, y_va, tag):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(Xn_tr.shape[1], CAT_CARDS_GLOBAL).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()

    xn_t, xc_t = to_tensors(Xn_tr, Xc_tr)
    ds = torch.utils.data.TensorDataset(xn_t, xc_t, torch.tensor(y_tr))
    xn_v, xc_v = to_tensors(Xn_va, Xc_va)
    g = torch.Generator().manual_seed(seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)

    t0 = time.time()
    best_auc, best_state, bad = 0.0, {}, 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb_n, xb_c, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb_n.to(DEVICE), xb_c.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            opt.step()
        va_auc = roc_auc_score(y_va, predict(model, xn_v, xc_v))
        eta = (time.time() - t0) / ep * (MAX_EPOCHS - ep)
        prog(f"{tag} epoch {ep}/{MAX_EPOCHS} val_auc={va_auc:.5f} "
             f"elapsed={time.time()-t0:.0f}s eta={eta:.0f}s")
        if va_auc > best_auc:
            best_auc, best_state, bad = va_auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                prog(f"{tag} early stop @epoch {ep}, best={best_auc:.5f}")
                break
    model.load_state_dict(best_state)
    return model, best_auc


CAT_CARDS_GLOBAL = None


def main():
    global CAT_CARDS_GLOBAL
    t_start = time.time()
    dev_name = torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu'
    print("=" * 60)
    print(f"DV-05: MLP diversity member | device={dev_name}")
    print("=" * 60)

    train, test = load_data()
    Xn_tr_all, Xc_tr_all, y, Xn_te, Xc_te, num_features, cat_cards = get_features(train, test)
    CAT_CARDS_GLOBAL = cat_cards
    prog(f"data ready: train={len(y)} test={len(Xn_te)} num_feats={len(num_features)} cards={cat_cards}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    oof_seeds, test_seeds = [], []
    for s_i, seed in enumerate(range(SEED, SEED + N_SEEDS)):
        oof = np.zeros(len(y))
        test_pred = np.zeros(len(Xn_te))
        for f_i, (tr_idx, va_idx) in enumerate(splits):
            sc = StandardScaler().fit(Xn_tr_all.iloc[tr_idx])
            model, best_auc = train_fold(
                seed,
                sc.transform(Xn_tr_all.iloc[tr_idx]), Xc_tr_all.iloc[tr_idx], y[tr_idx],
                sc.transform(Xn_tr_all.iloc[va_idx]), Xc_tr_all.iloc[va_idx], y[va_idx],
                tag=f"seed {s_i+1}/{N_SEEDS} fold {f_i+1}/{N_FOLDS}")
            oof[va_idx] = predict(model, *to_tensors(sc.transform(Xn_tr_all.iloc[va_idx]),
                                                     Xc_tr_all.iloc[va_idx]))
            test_pred += predict(model, *to_tensors(sc.transform(Xn_te), Xc_te)) / N_FOLDS
        cv_auc = roc_auc_score(y, oof)
        prog(f"seed {s_i+1}/{N_SEEDS} DONE oof_auc={cv_auc:.5f} total_elapsed={time.time()-t_start:.0f}s")
        oof_seeds.append(oof)
        test_seeds.append(test_pred)

    oof_final = np.mean(oof_seeds, axis=0)
    test_final = np.clip(np.mean(test_seeds, axis=0), 0, 1)
    cv_auc = roc_auc_score(y, oof_final)

    # Self-checks: fail loudly before saving anything
    assert not np.isnan(oof_final).any() and not np.isnan(test_final).any(), "NaN in predictions"
    assert oof_final.min() >= 0 and oof_final.max() <= 1, "OOF out of [0,1]"
    assert len(test_final) == len(test) == 296302, "wrong test length"
    mean_gap = abs(test_final.mean() - pd.Series(y).mean())
    prog(f"FINAL oof_auc={cv_auc:.5f} test_mean={test_final.mean():.5f} "
         f"target_mean={pd.Series(y).mean():.5f} gap={mean_gap:.4f}")

    sub_id = pd.read_csv(DATA_DIR / 'sample_submission.csv')['id']
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sub_path = OUT_DIR / 'submission_dv05_mlp.csv'
    pd.DataFrame({'id': sub_id, 'addicted_label': test_final}).to_csv(sub_path, index=False)
    oof_path = OUT_DIR / 'dv05_oof.csv'
    pd.DataFrame({'id': train['id'], 'dv05_oof': oof_final}).to_csv(oof_path, index=False)

    exp_file = OUT_DIR / 'experiments.csv'
    row = pd.DataFrame({
        'method': ['mlp_5seed'], 'cv_auc': [cv_auc], 'exp_id': ['DV-05'],
        'category': ['Pool'], 'description': ['MLP 256-128-64, cat embeddings, 5-seed bagged'],
        'status': ['pending_LB']})
    pd.concat([pd.read_csv(exp_file), row], ignore_index=True).to_csv(exp_file, index=False) \
        if exp_file.exists() else row.to_csv(exp_file, index=False)

    prog(f"saved: {sub_path} | {oof_path} | wall_time={(time.time()-t_start)/60:.1f}min")


if __name__ == '__main__':
    main()
