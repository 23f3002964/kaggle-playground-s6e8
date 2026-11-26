"""
DV-11 on Kaggle: deeper search over 81-member public OOF pool + own MLP.
- uniform top-N sweep (fit-free -> global AUC honest)
- hill climb, ALL members as candidates, up to 80 slots
Writes submission only if nested CV beats naji18's 0.96986 (-> LB 0.97096).
Inputs: dataset adityaraj0612/s6e8-stack-pool + competition playground-series-s6e8
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

POOL = Path('/kaggle/input/s6e8-stack-pool')
OUT = Path('/kaggle/working')
SEED, N_FOLDS, MAX_STEPS, TOL = 42, 5, 80, 1e-7
BEST_KNOWN = 0.96986          # naji18 bands OOF -> LB 0.97096


def _find(name):
    hits = list(Path('/kaggle/input').rglob(name))
    assert hits, f"{name} not found anywhere under /kaggle/input: " \
                 f"{[str(p) for p in Path('/kaggle/input').iterdir()]}"
    return hits[0].parent


COMP = _find('train.csv')
LIB = _find('oof_tabm_seed3.npy')
GOLEM = _find('oof_a.npy')
OWN = _find('dv05_oof.csv')

t0 = time.time()
def prog(msg):
    print(f"[DV11] {msg} (+{time.time()-t0:.0f}s)", flush=True)


def fast_auc(y, s):
    r = rankdata(s)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def load_members(y_len, te_len):
    members = []
    for po in sorted(LIB.glob('oof_*.npy')):
        nm = po.name[4:-4]
        pt = po.parent / f'test_{nm}.npy'
        if pt.exists():
            o, t = np.load(po).ravel(), np.load(pt).ravel()
            assert len(o) == y_len and len(t) == te_len, f"{nm}: bad length"
            members.append((f'lib/{nm}', o, t))
    for pf in sorted(GOLEM.glob('test_*.npy')):
        nm = pf.stem[5:]
        o = np.load(GOLEM / f'oof_{nm}.npy').ravel()
        members.append((f'gol/{nm}', o, np.load(pf).ravel()))
    return members


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
    train = pd.read_csv(COMP / 'train.csv', usecols=['id', 'addicted_label'])
    y = train['addicted_label'].values.astype(np.int8)
    sub_ids = pd.read_csv(COMP / 'sample_submission.csv')['id']
    te_len = len(sub_ids)
    mem = load_members(len(y), te_len)

    dv05_oof = pd.read_csv(OWN / 'dv05_oof.csv')
    dv05_test = pd.read_csv(OWN / 'submission_dv05_mlp.csv')
    if dv05_oof['id'].equals(train['id']) and dv05_test['id'].equals(sub_ids):
        mem.append(('own/mlp', dv05_oof['dv05_oof'].values, dv05_test['addicted_label'].values))
        prog("own/mlp added")
    prog(f"members: {len(mem)}")

    n = len(y)
    R = np.column_stack([rankdata(o) / n for _, o, _ in mem]).astype(np.float32)
    T = np.column_stack([rankdata(t) / te_len for _, _, t in mem]).astype(np.float32)
    names = [nm for nm, *_ in mem]
    del mem

    singles = np.array([roc_auc_score(y, R[:, j]) for j in range(R.shape[1])])
    order = list(np.argsort(-singles))

    prog("--- uniform top-N sweep ---")
    best_n, best_auc_u = 0, 0.0
    for N in range(3, min(41, R.shape[1])):
        auc = roc_auc_score(y, R[:, order[:N]].mean(axis=1))
        if N <= 12 or auc > best_auc_u:
            prog(f"  top-{N:2d}: {auc:.5f}")
        if auc > best_auc_u:
            best_n, best_auc_u = N, auc
    prog(f"best uniform: top-{best_n} = {best_auc_u:.5f}")

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
    prog(f"NESTED deep-hc={auc_hc:.5f} | uniform={best_auc_u:.5f} | "
         f"beats naji18({BEST_KNOWN})? {max(auc_hc, best_auc_u) > BEST_KNOWN}")

    winner_is_hc = auc_hc >= best_auc_u
    if max(auc_hc, best_auc_u) > BEST_KNOWN:
        if winner_is_hc:
            counts, fit_auc = hill_climb(R, y, order)
            tot = sum(counts.values())
            blend = np.asarray(sum(c * T[:, j] for j, c in counts.items()) / tot)
            sel = {names[j]: c for j, c in counts.items() if c}
            tag = f'dv11_hc_{auc_hc:.5f}'
            prog(f"full-fit hc AUC={fit_auc:.5f} slots={tot}")
        else:
            blend = np.asarray(T[:, order[:best_n]].mean(axis=1))
            sel = {names[j]: 1 for j in order[:best_n]}
            tag = f'dv11_unif{best_n}_{best_auc_u:.5f}'
            prog("uniform top-N applied to test")
        out = OUT / f'submission_{tag}.csv'
        assert len(ids_ok := sub_ids) == len(blend) == 296302
        pd.DataFrame({'id': ids_ok, 'addicted_label': np.clip(blend, 0, 1)}).to_csv(out, index=False)
        prog(f"saved {out.name}")
        prog("weights: " + str(dict(sorted(sel.items(), key=lambda x: -x[1])[:15])))
    else:
        prog("no improvement over naji18 baseline — no submission written")


if __name__ == '__main__':
    main()
