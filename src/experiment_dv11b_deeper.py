"""
DV-11b: Deeper search over the same 81-member pool.
- uniform top-N sweep (fit-free, so global AUC is honest)
- hill climb with ALL members as candidates, up to 80 slots
Submits nothing unless nested CV beats naji18's 0.96986 (the score that
already mapped to LB 0.97096).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from experiment_dv11_pool_stack import BASE, LIB, GOLEM, SUBS, SEED, N_FOLDS, TOL, \
    load_members, fast_auc, hill_climb

MAX_STEPS = 80
BEST_KNOWN = 0.96986          # naji18 bands OOF -> LB 0.97096

t0 = time.time()
def prog(msg):
    print(f"[DV11b] {msg} (+{time.time()-t0:.0f}s)", flush=True)


def main():
    train = pd.read_csv(BASE / 'data/train.csv', usecols=['id', 'addicted_label'])
    y = train['addicted_label'].values.astype(np.int8)
    te_len = len(pd.read_csv(BASE / 'data/test.csv', usecols=['id']))
    mem = load_members(len(y), te_len)

    # add our MLP (has BOTH oof + test)
    dv05_oof = pd.read_csv(BASE / 'models/dv05_oof.csv')
    dv05_test = pd.read_csv(SUBS / 'submission_dv05_mlp.csv')
    sub_ids = pd.read_csv(BASE / 'data/sample_submission.csv')['id']
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

    # ---- fit-free uniform top-N sweep (global AUC is honest here) ----
    prog("--- uniform top-N sweep ---")
    best_n, best_auc = 0, 0.0
    for N in range(3, min(41, R.shape[1])):
        auc = roc_auc_score(y, R[:, order[:N]].mean(axis=1))
        if N <= 12 or auc > best_auc:
            prog(f"  top-{N:2d}: {auc:.5f}")
        if auc > best_auc:
            best_n, best_auc = N, auc
    prog(f"best uniform: top-{best_n} = {best_auc:.5f}")

    # ---- deep hill climb, nested ----
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    hc_oof = np.zeros(len(y))
    tot_slots = []
    for k, (fit_i, val_i) in enumerate(skf.split(R, y)):
        fi = np.sort(fit_i)
        counts, _ = hill_climb(R[fi], y[fi], order)
        tot = sum(counts.values())
        tot_slots.append(tot)
        for j, c in counts.items():
            if c:
                hc_oof[val_i] += c * R[val_i, j]
        hc_oof[val_i] /= tot
        del fi
        prog(f"fold {k+1}/{N_FOLDS}: slots={tot}")
    auc_hc = roc_auc_score(y, hc_oof)
    prog(f"NESTED deep-hc={auc_hc:.5f} (beats naji18 {BEST_KNOWN}? {auc_hc > BEST_KNOWN})")

    if auc_hc > best_auc:
        counts, fit_auc = hill_climb(R, y, order)
        tot = sum(counts.values())
        blend = np.asarray(sum(c * T[:, j] for j, c in counts.items()) / tot)
        sel = {names[j]: c for j, c in counts.items() if c}
        out = SUBS / f'submission_dv11b_hc_{auc_hc:.5f}.csv'
        ids = pd.read_csv(BASE / 'data/sample_submission.csv')['id']
        assert len(ids) == len(blend) == 296302
        pd.DataFrame({'id': ids, 'addicted_label': np.clip(blend, 0, 1)}).to_csv(out, index=False)
        prog(f"saved {out} | slots={tot} members={len(sel)}")
        prog("weights: " + str(dict(sorted(sel.items(), key=lambda x: -x[1]))))
    else:
        prog("no improvement over simple approaches — no submission written")


if __name__ == '__main__':
    main()
