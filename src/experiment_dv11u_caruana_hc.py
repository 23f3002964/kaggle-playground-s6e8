"""
DV-11 UPGRADE: Caruana Hill Climbing with Replacement + Negative Weights
Based on: Chris Deotte 1st place S5E5, Caruana et al. "Getting the Most Out of Ensemble Selection"
Key insight: Greedy HC with replacement allows same model multiple times (bagging effect) + negative weights for anti-correlation
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
SEED, N_FOLDS, MAX_STEPS, TOP_K, TOL = 42, 5, 100, 50, 1e-7

t0 = time.time()
def prog(msg):
    print(f"[DV11U] {msg} (+{time.time()-t0:.0f}s)", flush=True)


def fast_auc(y, s):
    """Exact AUC via mid-rank statistic (~40ms @550k rows)."""
    r = rankdata(s)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def load_members(y_len, te_len):
    """Return list of (name, oof_ranked, test_ranked)."""
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


def hill_climb_caruana(R_fit, y_fit, cand_indices, max_steps=100, allow_negative=True):
    """
    Caruana hill climbing with replacement and optional negative weights.
    Each step: add the member (or its negative) that most improves AUC.
    Replacement = same member can be added multiple times.
    Negative = member can be subtracted (weight -1) to exploit anti-correlation.
    """
    n_cand = len(cand_indices)
    counts = {j: 0 for j in cand_indices}
    neg_counts = {j: 0 for j in cand_indices} if allow_negative else {}
    s = np.zeros(len(y_fit))
    best = 0.5
    
    for step in range(max_steps):
        best_j, best_sign, best_auc = None, 1, best + TOL
        
        for j in cand_indices:
            # Try positive
            a_pos = fast_auc(y_fit, s + R_fit[:, j])
            if a_pos > best_auc:
                best_j, best_sign, best_auc = j, 1, a_pos
            
            # Try negative (if allowed)
            if allow_negative:
                a_neg = fast_auc(y_fit, s - R_fit[:, j])
                if a_neg > best_auc:
                    best_j, best_sign, best_auc = j, -1, a_neg
        
        if best_j is None:
            prog(f"  HC converged at step {step}, AUC={best:.6f}")
            break
        
        if best_sign == 1:
            s += R_fit[:, best_j]
            counts[best_j] += 1
        else:
            s -= R_fit[:, best_j]
            neg_counts[best_j] += 1
        best = best_auc
        
        if step % 10 == 0:
            pos_used = sum(1 for v in counts.values() if v > 0)
            neg_used = sum(1 for v in neg_counts.values() if v > 0) if allow_negative else 0
            prog(f"  step {step}: AUC={best:.6f}, pos_members={pos_used}, neg_members={neg_used}, total_slots={sum(counts.values()) + sum(neg_counts.values())}")
    
    return counts, neg_counts, best


def ridge_stack(R_fit, y_fit, R_eval, alphas):
    """Ridge on standardized ranks via normal equations."""
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

    # Spot-check
    for nm, expect in [('tabm_seed3', 0.96867), ('naji03', 0.96881), ('lookup', 0.96853)]:
        i = names.index(f'lib/{nm}')
        got = roc_auc_score(y, R[:, i])
        prog(f"check {nm}: {got:.5f} (manifest {expect})")
        assert abs(got - expect) < 0.001, "alignment broken"

    singles = np.array([roc_auc_score(y, R[:, j]) for j in range(R.shape[1])])
    top = list(np.argsort(-singles)[:TOP_K])
    prog("top single AUCs: " + ", ".join(f"{names[j]}={singles[j]:.5f}" for j in top[:10]))

    # Nested evaluation on frozen folds
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    unif_pred = R.mean(axis=1)
    auc_unif = roc_auc_score(y, unif_pred)
    prog(f"uniform all-{R.shape[1]}: AUC={auc_unif:.5f}")

    hc_oof = np.zeros(len(y))
    hc_neg_oof = np.zeros(len(y))
    ridge_oof = np.zeros(len(y))
    alphas = np.logspace(-3, 3, 13)
    
    for k, (fit_i, val_i) in enumerate(skf.split(R, y)):
        fit_sorted = np.sort(fit_i)
        val_sorted = np.sort(val_i)
        R_fit = R[fit_sorted]
        y_fit = y[fit_sorted]
        R_val = R[val_sorted]
        
        # Caruana HC with replacement + negative
        counts, neg_counts, _ = hill_climb_caruana(R_fit, y_fit, top, max_steps=MAX_STEPS, allow_negative=True)
        tot_pos = sum(counts.values())
        tot_neg = sum(neg_counts.values())
        tot = tot_pos + tot_neg
        
        if tot > 0:
            for j, c in counts.items():
                if c:
                    hc_oof[val_i] += c * R[val_i, j]
            for j, c in neg_counts.items():
                if c:
                    hc_neg_oof[val_i] -= c * R[val_i, j]
            hc_oof[val_i] /= tot
            hc_neg_oof[val_i] /= tot
        
        # Ridge stack
        ridge_oof[val_i] = ridge_stack(R_fit, y_fit, R_val, alphas)
        del R_fit
        prog(f"fold {k+1}/{N_FOLDS} done (hc slots={tot}, pos={tot_pos}, neg={tot_neg})")

    auc_hc = roc_auc_score(y, hc_oof)
    auc_hc_neg = roc_auc_score(y, hc_neg_oof)
    auc_ridge = roc_auc_score(y, ridge_oof)
    prog(f"NESTED scores: hc={auc_hc:.5f} hc_neg={auc_hc_neg:.5f} ridge={auc_ridge:.5f} uniform={auc_unif:.5f}")

    # Refit winners on ALL rows -> test submission
    results = {
        'hc': (auc_hc, lambda: hill_climb_caruana(R, y, top, max_steps=MAX_STEPS, allow_negative=False)),
        'hc_neg': (auc_hc_neg, lambda: hill_climb_caruana(R, y, top, max_steps=MAX_STEPS, allow_negative=True)),
        'ridge': (auc_ridge, None),
        'uniform': (auc_unif, None),
    }
    win = max(results, key=lambda k: results[k][0])
    prog(f"winner: {win} ({results[win][0]:.5f})")

    if win in ['hc', 'hc_neg']:
        allow_neg = win == 'hc_neg'
        counts, neg_counts, fit_auc = results[win][1]()
        tot = sum(counts.values()) + sum(neg_counts.values())
        prog(f"full-fit hc AUC={fit_auc:.5f}, slots={tot}, pos_members={sum(1 for v in counts.values() if v>0)}, neg_members={sum(1 for v in neg_counts.values() if v>0)}")
        
        blend_test = np.zeros(te_len)
        for j, c in counts.items():
            if c:
                blend_test += c * T[:, j]
        for j, c in neg_counts.items():
            if c:
                blend_test -= c * T[:, j]
        blend_test /= tot
        sel = {names[j]: c for j, c in counts.items() if c}
        if neg_counts:
            for j, c in neg_counts.items():
                if c:
                    sel[names[j] + '_NEG'] = -c
    elif win == 'ridge':
        blend_test = ridge_stack(R, y, T, alphas)
        sel = {}
    else:
        blend_test = T.mean(axis=1)
        sel = {}

    blend_test = np.asarray(blend_test)
    prog("top weights: " + str(dict(sorted(sel.items(), key=lambda x: -abs(x[1]))[:15])))

    out = SUBS / f'submission_dv11u_{win}_{results[win][0]:.5f}.csv'
    sub_ids = pd.read_csv(BASE / 'data/sample_submission.csv')['id']
    assert len(sub_ids) == len(blend_test) == 296302
    pd.DataFrame({'id': sub_ids, 'addicted_label': np.clip(blend_test, 0, 1)}).to_csv(out, index=False)
    prog(f"saved {out} mean={blend_test.mean():.5f}")

    exp = SUBS.parent / 'logs/experiments.csv'
    row = pd.DataFrame({'method': [f'dv11u_{win}'], 'cv_auc': [results[win][0]],
                        'exp_id': ['DV-11U'], 'category': ['Pool'],
                        'description': [f'{win} Caruana HC (replacement+neg) over {R.shape[1]} public OOF members (nested)'],
                        'status': ['pending_LB']})
    pd.concat([pd.read_csv(exp), row], ignore_index=True).to_csv(exp, index=False)
    prog("experiments.csv updated")


if __name__ == '__main__':
    main()