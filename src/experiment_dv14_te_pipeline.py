"""
DV-14: Exact-Value TE Pipeline — the missed lever
Community-verified findings implemented (tomasa2/kito_pl/echloeprice):
1. Stringified nested OOF target encoding on ALL columns (continuous included)
2. Transductive frequency encoding (train+test counts, no labels)
3. LGBM/XGB max_bin=2047 (distinct-value count, not default 255)
4. CatBoost builds its own ordered target statistics from raw string levels
5. Decimal lattice: first decimal digit of daily_screen_time_hours
6. Imputed columns as EXTRA features, NaN kept native
7. Generator invariant: other_screen = daily - social - gaming - work
Frozen folds: StratifiedKFold(5, shuffle=True, seed=42) on train.csv file order.
Outputs: OOF csv, test preds csv, 2 submission candidates, all AUCs.
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5
INNER_FOLDS = 10
SMOOTHING = 10

if os.path.exists('/kaggle/input/playground-series-s6e8/train.csv'):
    DATA_DIR = Path('/kaggle/input/playground-series-s6e8')
elif os.path.exists('/kaggle/input/competitions/playground-series-s6e8/train.csv'):
    DATA_DIR = Path('/kaggle/input/competitions/playground-series-s6e8')
elif os.path.exists('/kaggle/working/train.csv'):
    DATA_DIR = Path('/kaggle/working')
else:
    DATA_DIR = Path('../data')

OUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('../submissions')

NUM_COLS = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
            'work_study_hours', 'sleep_hours', 'notifications_per_day',
            'app_opens_per_day', 'weekend_screen_time']
CAT_COLS = ['gender', 'stress_level', 'academic_work_impact']
ALL_COLS = NUM_COLS + CAT_COLS


def prog(msg):
    print(f"[TE] {msg}", flush=True)


def stringify(s: pd.Series) -> pd.Series:
    # pandas-version-safe: explicit missing level, never silent NA drop
    return s.astype(object).fillna('__missing__').astype(str)


def smoothed_te_stats(levels: pd.Series, y: np.ndarray, smoothing: float, global_mean: float) -> pd.Series:
    df = pd.DataFrame({'lvl': levels.values, 'y': y})
    agg = df.groupby('lvl')['y'].agg(['mean', 'count'])
    return (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)


def main():
    t0 = time.time()
    prog("=" * 64)
    prog("DV-14 exact-value TE pipeline")
    prog("=" * 64)

    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    y = train['addicted_label'].values
    n_tr, n_te = len(train), len(test)
    train_ids = train['id'].values
    test_ids = pd.read_csv(DATA_DIR / 'sample_submission.csv')['id'].values
    prog(f"train {train.shape}, test {test.shape}, base rate {y.mean():.5f}")

    # ---- 1. Stringify all columns + transductive frequency (no labels) ----
    str_tr = pd.DataFrame(index=range(n_tr))
    str_te = pd.DataFrame(index=range(n_te))
    freq_tr = np.zeros((n_tr, len(ALL_COLS)))
    freq_te = np.zeros((n_te, len(ALL_COLS)))
    for j, c in enumerate(ALL_COLS):
        comb = stringify(pd.concat([train[c], test[c]], ignore_index=True))
        str_tr[c] = comb.iloc[:n_tr].values
        str_te[c] = comb.iloc[n_tr:].values
        counts = comb.value_counts()
        freq_tr[:, j] = comb.iloc[:n_tr].map(counts).values
        freq_te[:, j] = comb.iloc[n_tr:].map(counts).values
        assert str_tr[c].notna().all() and str_te[c].notna().all(), f"coverage loss in {c}"
        prog(f"  str+freq {c}: {comb.nunique()} levels")

    # ---- 2. Deterministic features (no target involved) ----
    medians = {c: train[c].median() for c in NUM_COLS}

    def det_feats(df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in NUM_COLS:
            cols.append(df[c].values.astype(np.float32))
            cols.append(df[c].fillna(medians[c]).values.astype(np.float32))
        d = df['daily_screen_time_hours']
        fd = np.where(d.isna().values, -1, ((d.values % 1) * 10).astype(int)).astype(np.float32)
        cols.append(fd)
        other = (df['daily_screen_time_hours'] - df['social_media_hours']
                 - df['gaming_hours'] - df['work_study_hours']).values.astype(np.float32)
        cols.append(other)
        return np.column_stack(cols)

    det_names = [f'{c}_raw' for c in NUM_COLS] + [f'{c}_imp' for c in NUM_COLS] \
        + ['daily_first_digit', 'other_screen']
    X_det_tr = det_feats(train)
    X_det_te = det_feats(test)
    freq_names = [f'{c}_freq' for c in ALL_COLS]
    prog(f"det features: {X_det_tr.shape[1]}, freq features: {len(freq_names)}")

    # ---- 3. Outer folds + nested TE + models ----
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + 1)

    oof_lgbm = np.zeros(n_tr); te_lgbm = np.zeros(n_te)
    oof_xgb = np.zeros(n_tr); te_xgb = np.zeros(n_te)
    oof_cb = np.zeros(n_tr); te_cb = np.zeros(n_te)

    cat_feat_names = [f'{c}_str' for c in ALL_COLS]

    def cb_frame(det: np.ndarray, strdf: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(det, columns=det_names)
        for c in ALL_COLS:
            df[f'{c}_str'] = strdf[c].values
        return df

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_tr), y)):
        tfold = time.time()
        global_mean = y[tr_idx].mean()

        # nested OOF target encoding per column
        te_tr = np.zeros((len(tr_idx), len(ALL_COLS)))
        te_va = np.zeros((len(va_idx), len(ALL_COLS)))
        te_te = np.zeros((n_te, len(ALL_COLS)))
        for j, c in enumerate(ALL_COLS):
            col_tr = str_tr[c].iloc[tr_idx]
            y_tr = y[tr_idx]
            # inner OOF for the outer-train rows themselves
            enc = np.full(len(tr_idx), global_mean)
            for itr, iva in inner.split(np.zeros(len(tr_idx)), y_tr):
                stats = smoothed_te_stats(col_tr.iloc[itr], y_tr[itr], SMOOTHING, global_mean)
                enc[iva] = col_tr.iloc[iva].map(stats).fillna(global_mean).values
            te_tr[:, j] = enc
            # val + test from full outer-train stats
            stats_full = smoothed_te_stats(col_tr, y_tr, SMOOTHING, global_mean)
            te_va[:, j] = str_tr[c].iloc[va_idx].map(stats_full).fillna(global_mean).values
            te_te[:, j] = str_te[c].map(stats_full).fillna(global_mean).values
        prog(f"fold {fold}: TE built in {time.time()-tfold:.0f}s")

        # View A (LGBM/XGB): det + freq + TE
        Xa_tr = np.hstack([X_det_tr[tr_idx], freq_tr[tr_idx], te_tr])
        Xa_va = np.hstack([X_det_tr[va_idx], freq_tr[va_idx], te_va])
        Xa_te = np.hstack([X_det_te, freq_te, te_te])
        feat_names = det_names + freq_names + [f'{c}_te' for c in ALL_COLS]

        # LGBM
        lgb_params = dict(objective='binary', metric='auc', boosting_type='gbdt',
                          num_leaves=63, learning_rate=0.03, feature_fraction=0.8,
                          bagging_fraction=0.8, bagging_freq=5, min_child_samples=50,
                          max_bin=2047, seed=SEED, n_jobs=-1, verbose=-1)
        dtr = lgb.Dataset(Xa_tr, label=y[tr_idx], feature_name=feat_names)
        dva = lgb.Dataset(Xa_va, label=y[va_idx], reference=dtr)
        m = lgb.train(lgb_params, dtr, num_boost_round=6000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        oof_lgbm[va_idx] = m.predict(Xa_va, num_iteration=m.best_iteration)
        te_lgbm += m.predict(Xa_te, num_iteration=m.best_iteration) / N_FOLDS
        prog(f"fold {fold}: lgbm best_iter={m.best_iteration} "
             f"valAUC={roc_auc_score(y[va_idx], oof_lgbm[va_idx]):.5f} [{time.time()-tfold:.0f}s]")
        del dtr, dva, m

        # XGB
        xgb_params = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist',
                          max_bin=2047, max_depth=6, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
                          seed=SEED, n_jobs=-1, verbosity=0)
        dtr = xgb.DMatrix(Xa_tr, label=y[tr_idx], feature_names=feat_names)
        dva = xgb.DMatrix(Xa_va, label=y[va_idx], feature_names=feat_names)
        m = xgb.train(xgb_params, dtr, num_boost_round=6000, evals=[(dva, 'valid')],
                      early_stopping_rounds=200, verbose_eval=0)
        oof_xgb[va_idx] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
        te_xgb += m.predict(xgb.DMatrix(Xa_te, feature_names=feat_names),
                            iteration_range=(0, m.best_iteration + 1)) / N_FOLDS
        prog(f"fold {fold}: xgb best_iter={m.best_iteration} "
             f"valAUC={roc_auc_score(y[va_idx], oof_xgb[va_idx]):.5f} [{time.time()-tfold:.0f}s]")
        del dtr, dva, m

        # CatBoost: raw string levels as cat_features (its own ordered TS)
        Xb_tr = cb_frame(X_det_tr[tr_idx], str_tr.iloc[tr_idx])
        Xb_va = cb_frame(X_det_tr[va_idx], str_tr.iloc[va_idx])
        Xb_te = cb_frame(X_det_te, str_te)
        m = CatBoostClassifier(iterations=1500, learning_rate=0.06, depth=7, l2_leaf_reg=3,
                               eval_metric='AUC', task_type='CPU', thread_count=-1,
                               random_seed=SEED, early_stopping_rounds=200, verbose=500)
        m.fit(Xb_tr, y[tr_idx], eval_set=(Xb_va, y[va_idx]), cat_features=cat_feat_names)
        oof_cb[va_idx] = m.predict_proba(Xb_va)[:, 1]
        te_cb += m.predict_proba(Xb_te)[:, 1] / N_FOLDS
        prog(f"fold {fold}: catboost best_iter={m.best_iteration_} "
             f"valAUC={roc_auc_score(y[va_idx], oof_cb[va_idx]):.5f} [{time.time()-tfold:.0f}s]")
        del m, Xb_tr, Xb_va, Xb_te

    # ---- 4. Blends: rank avg + logit stack ----
    def rankavg(*preds):
        return np.mean([rankdata(p) / len(p) for p in preds], axis=0)

    oof_rank = rankavg(oof_lgbm, oof_xgb, oof_cb)
    te_rank = rankavg(te_lgbm, te_xgb, te_cb)

    def logit(p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.clip(np.log(p / (1 - p)), -30, 30)

    S_tr = np.column_stack([logit(oof_lgbm), logit(oof_xgb), logit(oof_cb)])
    S_te = np.column_stack([logit(te_lgbm), logit(te_xgb), logit(te_cb)])
    oof_stack = np.zeros(n_tr); te_stack = np.zeros(n_te)
    for tr_idx, va_idx in skf.split(np.zeros(n_tr), y):
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(S_tr[tr_idx], y[tr_idx])
        oof_stack[va_idx] = lr.predict_proba(S_tr[va_idx])[:, 1]
        te_stack += lr.predict_proba(S_te)[:, 1] / N_FOLDS

    prog("=" * 64)
    prog(f"OOF AUC lgbm      : {roc_auc_score(y, oof_lgbm):.5f}")
    prog(f"OOF AUC xgb       : {roc_auc_score(y, oof_xgb):.5f}")
    prog(f"OOF AUC catboost  : {roc_auc_score(y, oof_cb):.5f}")
    prog(f"OOF AUC rankblend : {roc_auc_score(y, oof_rank):.5f}")
    prog(f"OOF AUC logitstack: {roc_auc_score(y, oof_stack):.5f}")
    prog("=" * 64)

    # ---- 5. Save everything ----
    pd.DataFrame({'id': train_ids, 'oof_lgbm': oof_lgbm, 'oof_xgb': oof_xgb,
                  'oof_cb': oof_cb, 'oof_stack': oof_stack}).to_csv(OUT_DIR / 'te_oof.csv', index=False)
    pd.DataFrame({'id': test_ids, 'lgbm': te_lgbm, 'xgb': te_xgb, 'cb': te_cb,
                  'stack': te_stack, 'rankblend': te_rank}).to_csv(OUT_DIR / 'te_test_preds.csv', index=False)
    pd.DataFrame({'id': test_ids, 'addicted_label': te_rank}).to_csv(
        OUT_DIR / 'submission_te_rankblend.csv', index=False)
    pd.DataFrame({'id': test_ids, 'addicted_label': te_stack}).to_csv(
        OUT_DIR / 'submission_te_stack.csv', index=False)
    prog(f"saved to {OUT_DIR}; total {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
