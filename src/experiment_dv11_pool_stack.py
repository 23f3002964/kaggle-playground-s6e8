"""
DV-11: Rank-space hill climbing + ridge stack over the public OOF library.
Members: szymonkapiski/s6e8-oof-library-47-models (74) + golem honest 7 = 81,
all aligned to the frozen StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
split over train.csv in FILE ORDER. Aggregators are nested-evaluated
(fit on 4 folds -> score held-out fold) so the CV number is honest, no ratta.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent
LIB = BASE / 'data/pool/oof_library/oof'
GOLEM = BASE / 'data/pool/golem'
SUBS = BASE / 'submissions'
SEED, N_FOLDS, MAX_STEPS, TOP_K, TOL = 42, 5, 50, 40, 1e-7

t0 = time.time()
def prog(msg):
    print(f"[DV11] {msg} (+{time.time()-t0:.0f}s)", flush=True)


def fast_auc(y, s):
    """Exact AUC via mid-rank statistic (~40ms @550k rows)."""
    r = rankdata(s)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def load_members(y_len, te_len):
    """Return list of (name, oof_ranked, test_ranked). Ranks are monotone ->
    subsetting preserves order, so global ranking leaks nothing."""
    names = sorted(p.name[4:-4] for p in LIB.glob('oof_*.npy'))
    members = []
    for nm in names:
        po, pt = LIB / f'oof_{nm}.npy', LIB / f'test_{nm}.npy'
        if not (po.exists() and pt.exists()):
            continue
        o, t = np.load(po).ravel(), np.load(pt).ravel()
        assert len(o) == y_len and len(t) == te_len, f"{nm}: bad length"
        members.append((f'lib/{nm}', o, t))
    for f in sorted(GOLEM.glob('test_*.npy')):
        nm = f.stem[5:]
        o = np.load(GOLEM / f'oof_{nm}.npy').ravel()
        members.append((f'gol/{nm}', o, np.load(f).ravel()))
    return members


def ridge_stack(R_fit, y_fit, R_eval, alphas):
    """Ridge on standardized ranks via normal equations (memory-light).
    Alpha chosen by train-fold AUC only -> held-out rows stay honest."""
    R_fit = R_fit.copy()
    mu, sd = R_fit.mean(0), R_fit.std(0) + 1e-9
    R_fit -= mu
    R_fit /= sd
    yc = y_fit - y_fit.mean()
    G = R_fit.T @ R_fit
    b = R_fit.T @ yc
    eye = np.eye(R_fit.shape[1], dtype=np.float32)
    best = (-1.0, None)
    for a in alphas:
        w = np.linalg.solve(G + np.float32(a) * eye, b)
        auc = fast_auc(y_fit, R_fit @ w)
        if auc > best[0]:
            best = (auc, w)
    _, w = best
    return (R_eval - mu) / sd @ w


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


def main():
    train = pd.read_csv(BASE / 'data/train.csv', usecols=['id', 'addicted_label'])
    y = train['addicted_label'].values.astype(np.int8)
    te_len = len(pd.read_csv(BASE / 'data/test.csv', usecols=['id']))
    prog(f"train={len(y)} test={te_len}")

    mem = load_members(len(y), te_len)
    prog(f"members loaded: {len(mem)}")
    n, m = len(y), len(mem)
    R = np.empty((n, m), dtype=np.float32)
    T = np.empty((te_len, m), dtype=np.float32)
    names = []
    for i, (nm, o, t) in enumerate(mem):
        R[:, i] = rankdata(o) / n
        T[:, i] = rankdata(t) / te_len
        names.append(nm)
    del mem

    # spot-check two manifest AUCs to confirm alignment
    for nm, expect in [('tabm_seed3', 0.96867), ('naji03', 0.96881)]:
        i = names.index(f'lib/{nm}')
        got = roc_auc_score(y, R[:, i])
        prog(f"check {nm}: {got:.5f} (manifest {expect})")
        assert abs(got - expect) < 0.001, "alignment broken"

    singles = np.array([roc_auc_score(y, R[:, j]) for j in range(R.shape[1])])
    top = list(np.argsort(-singles)[:TOP_K])
    prog("top single AUCs: " + ", ".join(f"{names[j]}={singles[j]:.5f}" for j in top[:5]))

    # ---- nested evaluation on frozen folds ----
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    unif_pred = R.mean(axis=1)
    auc_unif = roc_auc_score(y, unif_pred)
    prog(f"uniform all-{R.shape[1]}: AUC={auc_unif:.5f}")

    hc_oof = np.zeros(len(y))
    ridge_oof = np.zeros(len(y))
    alphas = np.logspace(-3, 3, 13)
    for k, (fit_i, val_i) in enumerate(skf.split(R, y)):
        fit_sorted = np.sort(fit_i)
        R_fit = R[fit_sorted]
        counts, _ = hill_climb(R_fit, y[fit_i], top)
        tot = sum(counts.values())
        for j, c in counts.items():
            if c:
                hc_oof[val_i] += c * R[val_i, j]
        hc_oof[val_i] /= tot
        ridge_oof[val_i] = ridge_stack(R_fit, y[fit_sorted], R[np.sort(val_i)], alphas)
        del R_fit
        prog(f"fold {k+1}/{N_FOLDS} done (hc slots={tot})")

    auc_hc = roc_auc_score(y, hc_oof)
    auc_ridge = roc_auc_score(y, ridge_oof)
    prog(f"NESTED scores: hc={auc_hc:.5f} ridge={auc_ridge:.5f} uniform={auc_unif:.5f}")

    # ---- refit winner on ALL rows -> test submission ----
    results = {'hc': auc_hc, 'ridge': auc_ridge, 'uniform': auc_unif}
    win = max(results, key=lambda k: results[k])
    sel: dict = {}
    prog(f"winner: {win}")
    if win == 'hc':
        final_counts, fit_auc = hill_climb(R, y, top)
        tot = sum(final_counts.values())
        blend_test = sum(c * T[:, j] for j, c in final_counts.items()) / tot
        sel = {names[j]: c for j, c in final_counts.items() if c}
        prog(f"full-fit hc AUC={fit_auc:.5f}, slots={tot}, members used={len(sel)}")
    elif win == 'ridge':
        blend_test = ridge_stack(R, y, T, alphas)
        prog("ridge stack applied to test")
    else:
        blend_test = T.mean(axis=1)
    blend_test = np.asarray(blend_test)
    prog("top weights: " + str(dict(sorted(sel.items(), key=lambda x: -abs(x[1]))[:10])))

    out = SUBS / f'submission_dv11_{win}_{results[win]:.5f}.csv'
    sub_ids = pd.read_csv(BASE / 'data/sample_submission.csv')['id']
    assert len(sub_ids) == len(blend_test) == 296302
    pd.DataFrame({'id': sub_ids,
                  'addicted_label': np.clip(blend_test, 0, 1)}).to_csv(out, index=False)
    prog(f"saved {out} mean={blend_test.mean():.5f}")

    exp = SUBS.parent / 'logs/experiments.csv'
    row = pd.DataFrame({'method': [f'dv11_{win}'], 'cv_auc': [results[win]],
                        'exp_id': ['DV-11'], 'category': ['Pool'],
                        'description': [f'{win} over {R.shape[1]} public OOF members (nested)'],
                        'status': ['pending_LB']})
    pd.concat([pd.read_csv(exp), row], ignore_index=True).to_csv(exp, index=False)
    prog("experiments.csv updated")


if __name__ == '__main__':
    main()
