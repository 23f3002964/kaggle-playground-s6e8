"""
DV-01/DV-02/DV-03: Pool refresh + prune + aggregator test
Refresh the 13-Aug pool141 with new public members (najiama 15, anhad 20,
mohan 3, qamrodz 1) + our own v5/pl01 outputs.

Steps:
 1. Load all candidate TEST prediction files.
 2. Pairwise Spearman on ranks -> prune near-duplicates (>0.999).
 3. Aggregators in RANK space (AUC only cares about order):
    a. uniform rank-avg of pruned pool            (pool141's proven method)
    b. hill-climb (Caruana, with replacement) on members with OOF
    c. uniform rank-avg + own models at 5% weight (DV-04 pre-test)
 4. Report OOF AUC where measurable; write probe submissions.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

POOL = Path('/home/adi/CODE/Kaggle/playground-series-s6e8/data/pool')
SUBS = Path('/home/adi/CODE/Kaggle/playground-series-s6e8/submissions')
TRAIN = pd.read_csv('/home/adi/CODE/Kaggle/playground-series-s6e8/data/train.csv',
                    usecols=['id', 'addicted_label'])
y = TRAIN.set_index('id')['addicted_label']

def rank_norm(v):
    return rankdata(v, method='average') / len(v)

def load_test_pred(path, name):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    pred_col = 'addicted_label' if 'addicted_label' in df.columns else df.columns[1]
    s = df.set_index('id')[pred_col]
    assert len(s) == 296302, f"{name}: {len(s)} rows"
    return s

# ---- 1. Load candidates ----
members = {}   # name -> test preds (pd.Series indexed by id)
oofs = {}      # name -> OOF preds (pd.Series indexed by train id)

for f in sorted((POOL / 'anhad').glob('*.csv')):
    members[f'anhad/{f.stem}'] = load_test_pred(f, f.stem)

naji = {
    p.stem.replace('_submission', ''): p
    for p in (POOL / 'najiama_oof').glob('*_submission*.csv')
}
for stem, p in naji.items():
    members[f'naji/{stem}'] = load_test_pred(p, stem)
    oof_f = POOL / 'najiama_oof' / f'{stem}_oof_predictions.csv'
    if oof_f.exists():
        df = pd.read_csv(oof_f)
        df.columns = [c.lower() for c in df.columns]
        oofs[f'naji/{stem}'] = df.set_index('id')[df.columns[-1]]

MOHAN = [('mohan/dart', 'mohan_dart', 'submission_lgb_v3.csv', 'oof_lgb_v3.npy'),
         ('mohan/xgb', 'mohan_xgb', 'submission_xgb_v3.csv', 'oof_xgb_v3.npy'),
         ('mohan/catmlp', 'mohan_catmlp', 'submission_cat_v3.csv', 'oof_cat_v3.npy')]
for tag, d, sub, oof_f in MOHAN:
    members[tag] = load_test_pred(POOL / d / sub, tag)

train_ids = TRAIN['id'].values
for tag, d, sub, oof_f in MOHAN:
    p = POOL / d / oof_f
    if p.exists():
        oofs[tag] = pd.Series(np.load(p).ravel(), index=train_ids)

members['qamrodz'] = load_test_pred(POOL / 'qamrodz' / 'submission.csv', 'qamrodz')

members['own/v5_lgbmeta'] = pd.read_csv(
    SUBS / 'submission_pool141_improved_v5_lgbmeta.csv').set_index('id')['addicted_label']
members['own/pl01'] = pd.read_csv(
    SUBS / 'submission_pl01_v3.csv').set_index('id')['addicted_label']

print(f"Candidates: {len(members)}  |  with OOF: {len(oofs)}")

# sanity: all OOF AUCs
print("\n--- OOF AUCs ---")
for name, oof in sorted(oofs.items()):
    try:
        print(f"  {name:24s} {roc_auc_score(y.loc[oof.index], oof.values):.5f}")
    except Exception as e:
        print(f"  {name:24s} OOF align fail: {e}")

# ---- 2. Rank-space representation + duplicate pruning ----
names = list(members)
R = pd.DataFrame({n: rank_norm(members[n].values) for n in names},
                 index=members[names[0]].index)

corr = R.corr(method='spearman')
keep, dropped = [], []
for n in names:
    if any(corr.loc[n, k] > 0.999 for k in keep):
        dropped.append(n)
    else:
        keep.append(n)
print(f"\nPruned {len(dropped)} near-duplicates (>0.999): {dropped}")

# ---- 3a. Uniform rank-avg (pruned) ----
unif = R[keep].mean(axis=1)
sub_unif = pd.DataFrame({'id': unif.index, 'addicted_label': unif.values})
sub_unif.to_csv(SUBS / 'submission_dv01_refresh_unif.csv', index=False)
print(f"\n[3a] uniform rank-avg of {len(keep)} pruned members -> submission_dv01_refresh_unif.csv")

# ---- 3b. Hill climb on OOF-bearing members ----
hc_names = [n for n in keep if n in oofs]
print(f"\n[3b] hill climbing over {len(hc_names)} OOF members...")
O = pd.DataFrame({n: oofs[n].loc[train_ids].values for n in hc_names})
yv = y.loc[train_ids].values
Or = O.rank() / len(O)

sel, cur = [], np.zeros(len(O))
best_auc = 0.5
for it in range(60):
    gains = {n: roc_auc_score(yv, cur + Or[n]) for n in hc_names if n not in [] }
    n_best = max(gains, key=gains.get)
    if gains[n_best] <= best_auc + 1e-6:
        break
    sel.append(n_best)
    cur = cur + Or[n_best]
    best_auc = gains[n_best]
    if it < 10 or it % 10 == 0:
        print(f"  iter {it+1}: +{n_best:24s} AUC {best_auc:.5f}")
from collections import Counter
w = Counter(sel)
print(f"  HC selected {len(sel)} slots over {len(w)} members; AUC {best_auc:.5f}")
hc_test = sum(w[n] * R[n] for n in w) / len(sel)
sub_hc = pd.DataFrame({'id': hc_test.index, 'addicted_label': hc_test.values})
sub_hc.to_csv(SUBS / 'submission_dv02_hc.csv', index=False)
print(f"  -> submission_dv02_hc.csv")

# ---- 3c. Trimmed mean (DV-03 aggregator probe) ----
K = R[keep]
arr = np.sort(K.values, axis=1)
trimmed = arr[:, 1:-1].mean(axis=1)
pd.DataFrame({'id': K.index, 'addicted_label': trimmed}).to_csv(
    SUBS / 'submission_dv03_trimmed.csv', index=False)
print(f"\n[3c] trimmed mean (drop per-row min/max) over {len(keep)} members -> submission_dv03_trimmed.csv")

# ---- 3d. Pool + own injection 5% (DV-04 pretest) ----
own = ['own/v5_lgbmeta', 'own/pl01']
inj = (0.95 * unif + 0.05 * R[own].mean(axis=1))
pd.DataFrame({'id': inj.index, 'addicted_label': inj.values}).to_csv(
    SUBS / 'submission_dv04_inject5.csv', index=False)
print(f"\n[3c] 95% pool + 5% own -> submission_dv04_inject5.csv")

# pairwise corr summary of kept members vs own
for o in own:
    cs = corr.loc[o, keep].drop(o).sort_values(ascending=False)
    print(f"  {o} max-corr with pool: {cs.max():.4f} ({cs.index[0]}), median {cs.median():.4f}")
