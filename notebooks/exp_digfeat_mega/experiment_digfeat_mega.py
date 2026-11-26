"""
DIG-FEAT + MEGA STACK (one kernel, two stages):
Stage 1: Train LGBM+XGB with quantization-digit features (+pool141 interactions)
         on the frozen folds -> honest OOF (ES on inner 8%) + test preds.
Stage 2: Hill-climb over lib74 + golem7 + own MLP + new digit models.
         Submission written ONLY if nested CV beats naji18's 0.96986.
"""
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

OUT = Path('/kaggle/working')
SEED, N_FOLDS, MAX_STEPS, TOL = 42, 5, 80, 1e-7
BEST_KNOWN = 0.96986          # naji18 bands OOF -> LB 0.97096
HAS_GPU = shutil.which('nvidia-smi') is not None


def resolve_paths():
    comp = next(Path('/kaggle/input').rglob('train.csv')).parent
    pool = next(Path('/kaggle/input').rglob('oof_tabm_seed3.npy')).parent
    golem = next(Path('/kaggle/input').rglob('oof_a.npy')).parent
    own = next(Path('/kaggle/input').rglob('dv05_oof.csv')).parent
    return comp, pool, golem, own

t0 = time.time()
def prog(msg):
    print(f"[DIG] {msg} (+{time.time()-t0:.0f}s)", flush=True)


def fast_auc(y, s):
    r = rankdata(s)
    n_pos = int(y.sum())
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(y) - n_pos))


def hill_climb(R_fit, y_fit, cand):
    counts = {j: 0 for j in cand}
    s = np.zeros(len(y_fit))
    best = 0.5
    for _ in range(MAX_STEPS):
        best_j, best_auc = None, best + TOL
        for j in cand:
            a = fast_auc(y_fit, s + R_fit[:, j])
            if a > best_auc:
                best_j, best_auc = j, a
        if best_j is None:
            break
        s += R_fit[:, best_j]
        counts[best_j] += 1
        best = best_auc
    return counts, best


def add_features(df, num_cols):
    """pool141 interactions + quantization-digit features."""
    df['total_screen_time'] = df['daily_screen_time_hours'] + df['weekend_screen_time']
    df['screen_per_notification'] = df['daily_screen_time_hours'] / (df['notifications_per_day'] + 1e-6)
    df['screen_per_app_open'] = df['daily_screen_time_hours'] / (df['app_opens_per_day'] + 1e-6)
    df['notif_per_app_open'] = df['notifications_per_day'] / (df['app_opens_per_day'] + 1e-6)
    df['leisure_hours'] = df['social_media_hours'] + df['gaming_hours']
    df['leisure_productive_ratio'] = df['leisure_hours'] / (df['work_study_hours'] + 1e-6)
    df['sleep_debt'] = np.maximum(8 - df['sleep_hours'], 0)
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 50, 100],
                             labels=[0, 1, 2, 3, 4]).astype(float)
    stress_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Unknown': 1}
    impact_map = {'No': 0, 'Yes': 1, 'Unknown': 0}
    for col, mapping in [('stress_level', stress_map), ('academic_work_impact', impact_map)]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col].map(mapping), errors='coerce').fillna(1)
    # rounding patterns leak synthetic-generator signal
    for c in num_cols:
        x = pd.to_numeric(df[c], errors='coerce').to_numpy(dtype=float)
        nan = np.isnan(x)
        base = np.where(nan, 0.0, x - np.floor(x))
        frac = np.where(nan, -1.0, base)
        df[c + '_frac'] = frac
        df[c + '_d1'] = np.where(nan, -1, np.floor(base * 10 + 1e-9)).astype(int)
        df[c + '_d2'] = np.where(nan, -1, np.floor(base * 100 % 10 + 1e-9)).astype(int)
        df[c + '_is_int'] = np.where(nan, -1, (x == np.round(x)).astype(int)).astype(int)
    return df


def stage1(train, test, y):
    num_cols = [c for c in test.select_dtypes(include=[np.number]).columns if c != 'id']
    prog("filling missing + building features...")
    medians = {c: train[c].median() for c in num_cols}
    for d in (train, test):
        for c in num_cols:
            d[c] = d[c].fillna(medians[c])
        for c in ['gender', 'stress_level', 'academic_work_impact']:
            d[c] = d[c].fillna('Unknown')
    train = add_features(train, num_cols)
    test = add_features(test, num_cols)
    vals = sorted(train['gender'].unique())
    gm = {v: i for i, v in enumerate(vals)}
    train['gender'] = train['gender'].map(gm)
    test['gender'] = test['gender'].map(gm).fillna(0).astype(int)

    feats = [c for c in train.columns if c not in ('id', 'addicted_label')]
    X, Xt = train[feats], test[feats]
    prog(f"features={len(feats)} | gpu={HAS_GPU}")

    import lightgbm as lgb
    import xgboost as xgb

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    out = {}
    for name in ['lgb_digit', 'xgb_digit']:
        oof = np.zeros(len(y))
        tpred = np.zeros(len(Xt))
        for k, (fit_i, val_i) in enumerate(skf.split(X, y)):
            fi, vi = np.sort(fit_i), np.sort(val_i)
            Xf, yf = X.iloc[fi], y[fi]
            fp, ep, yp, ye = train_test_split(Xf, yf, test_size=0.08,
                                              random_state=SEED + k, stratify=yf)
            if name == 'lgb_digit':
                params = dict(objective='binary', metric='auc', num_leaves=63,
                              learning_rate=0.05, feature_fraction=0.8,
                              bagging_fraction=0.8, bagging_freq=5, seed=SEED,
                              verbose=-1, n_jobs=-1)
                m0 = lgb.train(params, lgb.Dataset(fp, label=yp), num_boost_round=2000,
                               valid_sets=[lgb.Dataset(ep, label=ye)],
                               callbacks=[lgb.early_stopping(50, verbose=False)])
                model = lgb.train(params, lgb.Dataset(Xf, label=yf),
                                  num_boost_round=max(int(m0.best_iteration * 1.05), 100))
                oof[vi] = model.predict(X.iloc[vi])
                tpred += model.predict(Xt) / N_FOLDS
            else:
                params = dict(objective='binary:logistic', eval_metric='auc',
                              max_depth=5, learning_rate=0.099, subsample=0.97,
                              colsample_bytree=0.66, min_child_weight=8,
                              reg_alpha=5.9, reg_lambda=0.2, gamma=0.015,
                              random_state=SEED, verbosity=0, n_jobs=-1,
                              tree_method='hist')
                if HAS_GPU:
                    params['device'] = 'cuda'
                m0 = xgb.train(params, xgb.DMatrix(fp, label=yp), num_boost_round=3000,
                               evals=[(xgb.DMatrix(ep, label=ye), 'es')],
                               early_stopping_rounds=50, verbose_eval=False)
                bi = m0.best_iteration + 1
                model = xgb.train(params, xgb.DMatrix(Xf, label=yf),
                                  num_boost_round=max(int(bi * 1.05), 100))
                rng = (0, max(int(bi * 1.05), 100))
                oof[vi] = model.predict(xgb.DMatrix(X.iloc[vi]), iteration_range=rng)
                tpred += model.predict(xgb.DMatrix(Xt), iteration_range=rng) / N_FOLDS
            prog(f"{name} fold {k+1}/{N_FOLDS} done")
        auc = roc_auc_score(y, oof)
        prog(f"{name}: honest OOF={auc:.5f}")
        out[name] = (oof, np.clip(tpred, 0, 1), auc)

    np.save(OUT / 'oof_lgb_digit.npy', out['lgb_digit'][0])
    np.save(OUT / 'test_lgb_digit.npy', out['lgb_digit'][1])
    np.save(OUT / 'oof_xgb_digit.npy', out['xgb_digit'][0])
    np.save(OUT / 'test_xgb_digit.npy', out['xgb_digit'][1])
    return out


def main():
    comp, pool, golem, own_ds = resolve_paths()
    train = pd.read_csv(comp / 'train.csv')
    sub_ids = pd.read_csv(comp / 'sample_submission.csv')['id']
    y = train['addicted_label'].values.astype(np.int8)
    test = pd.read_csv(comp / 'test.csv')

    dig = stage1(train, test, y)

    prog("--- stage 2: mega stack ---")
    mem = []
    for po in sorted(pool.glob('oof_*.npy')):
        nm = po.name[4:-4]
        pt = po.parent / f'test_{nm}.npy'
        if pt.exists():
            mem.append((f'lib/{nm}', np.load(po).ravel(), np.load(pt).ravel()))
    for pf in sorted(golem.glob('test_*.npy')):
        nm = pf.stem[5:]
        mem.append((f'gol/{nm}', np.load(golem / f'oof_{nm}.npy').ravel(), np.load(pf).ravel()))
    dv_oof = pd.read_csv(own_ds / 'dv05_oof.csv')
    dv_test = pd.read_csv(own_ds / 'submission_dv05_mlp.csv')
    mem.append(('own/mlp', dv_oof['dv05_oof'].values, dv_test['addicted_label'].values))
    mem.append(('own/lgb_digit', dig['lgb_digit'][0], dig['lgb_digit'][1]))
    mem.append(('own/xgb_digit', dig['xgb_digit'][0], dig['xgb_digit'][1]))
    prog(f"total members: {len(mem)}")

    te_len = len(sub_ids)
    R = np.column_stack([rankdata(o) / len(y) for _, o, _ in mem]).astype(np.float32)
    T = np.column_stack([rankdata(t) / te_len for _, _, t in mem]).astype(np.float32)
    names = [nm for nm, *_ in mem]
    del mem

    singles = np.array([roc_auc_score(y, R[:, j]) for j in range(R.shape[1])])
    order = list(np.argsort(-singles))

    best_n, best_u = 0, 0.0
    for N in range(3, 41):
        auc = roc_auc_score(y, R[:, order[:N]].mean(axis=1))
        if auc > best_u:
            best_n, best_u = N, auc
    li, xi = names.index('own/lgb_digit'), names.index('own/xgb_digit')
    prog(f"best uniform: top-{best_n}={best_u:.5f} | own singles: "
         f"lgb={singles[li]:.5f} xgb={singles[xi]:.5f}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    hc_oof = np.zeros(len(y))
    for k, (fit_i, val_i) in enumerate(skf.split(R, y)):
        fi = np.sort(fit_i)
        counts, _ = hill_climb(R[fi], y[fi], order)
        tot = sum(counts.values())
        for j, c in counts.items():
            if c:
                hc_oof[val_i] += c * R[val_i, j]
        hc_oof[val_i] /= tot
        del fi
        prog(f"fold {k+1}/{N_FOLDS}: slots={tot}")
    auc_hc = roc_auc_score(y, hc_oof)
    best_cv = max(auc_hc, best_u)
    prog(f"NESTED: hc={auc_hc:.5f} uniform={best_u:.5f} | beats {BEST_KNOWN}? {best_cv > BEST_KNOWN}")

    if best_cv > BEST_KNOWN:
        if auc_hc >= best_u:
            counts, _ = hill_climb(R, y, order)
            blend = np.asarray(sum(c * T[:, j] for j, c in counts.items()) / sum(counts.values()))
            sel = {names[j]: c for j, c in counts.items() if c}
            tag = f'dv12_hc_{auc_hc:.5f}'
        else:
            blend = np.asarray(T[:, order[:best_n]].mean(axis=1))
            sel = {names[j]: 1 for j in order[:best_n]}
            tag = f'dv12_unif{best_n}_{best_u:.5f}'
        assert len(sub_ids) == len(blend) == 296302
        out = OUT / f'submission_{tag}.csv'
        pd.DataFrame({'id': sub_ids,
                      'addicted_label': np.clip(blend, 0, 1)}).to_csv(out, index=False)
        prog(f"SAVED {out.name}")
        prog("weights: " + str(dict(sorted(sel.items(), key=lambda x: -x[1])[:15])))
    else:
        prog("no submission written — threshold not beaten")
        np.save(OUT / 'hc_oof.npy', hc_oof)


if __name__ == '__main__':
    main()
