"""
NAJI-BOOST: base = naji18 blend (honest OOF 0.96986 -> LB 0.97096).
Sweep tiny weights w of our new digit models on top:
    blend = (1-w)*rank(naji18) + w*rank(member)
All blends are fit-free, so global OOF AUC is honest.
Submission written ONLY if best OOF beats 0.96986.
Inputs: dataset s6e8-stack-pool (v2, has naji/), kernel output of
dv12-digit-mega-stack (oof/test_xgb_digit.npy etc), competition data.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

OUT = Path('/kaggle/working')
BEST_KNOWN = 0.96986
WS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]

t0 = time.time()
def prog(msg):
    print(f"[NB] {msg} (+{time.time()-t0:.0f}s)", flush=True)


def find(name):
    hits = list(Path('/kaggle/input').rglob(name))
    assert hits, f"{name} not found"
    return hits[0]


def main():
    comp = find('train.csv').parent
    train = pd.read_csv(comp / 'train.csv', usecols=['id', 'addicted_label'])
    y = train['addicted_label'].values.astype(np.int8)
    sub_ids = pd.read_csv(comp / 'sample_submission.csv')['id']

    nj_oof_df = pd.read_csv(find('18_blend_oof_predictions.csv'))
    nj_oof_df.columns = ['id', 'pred']
    assert nj_oof_df['id'].equals(train['id']), "naji oof id misalignment"
    nj_test_df = pd.read_csv(find('18_blend_submission.csv'))
    nj_test_df.columns = [c.lower() for c in nj_test_df.columns]
    assert len(nj_test_df) == len(sub_ids) and nj_test_df['id'].equals(sub_ids)

    Ro_naji = rankdata(nj_oof_df['pred'].values) / len(y)
    Tt_naji = rankdata(nj_test_df['addicted_label'].values) / len(sub_ids)
    base_auc = roc_auc_score(y, Ro_naji)
    prog(f"naji18 base OOF={base_auc:.5f} (expected ~0.96986)")

    members = []
    for tag in ['xgb_digit', 'lgb_digit']:
        oo, tt = find(f'oof_{tag}.npy'), find(f'test_{tag}.npy')
        o, t = np.load(oo).ravel(), np.load(tt).ravel()
        assert len(o) == len(y) and len(t) == len(sub_ids)
        auc = roc_auc_score(y, rankdata(o) / len(y))
        prog(f"member {tag}: single OOF={auc:.5f}")
        members.append((tag, rankdata(o) / len(y), rankdata(t) / len(sub_ids)))

    # also try the pair averaged
    pair_oof = (members[0][1] + members[1][1]) / 2
    pair_test = (members[0][2] + members[1][2]) / 2
    members.append(('digit_pair_avg', pair_oof, pair_test))

    best = (base_auc, None, None, None)   # auc, name, w, test_blend
    for tag, ro, tt in members:
        for w in WS:
            auc = roc_auc_score(y, (1 - w) * Ro_naji + w * ro)
            flag = ' *' if auc > best[0] else ''
            prog(f"  naji18 + {int(w*100):2d}% {tag:14s}: {auc:.5f}{flag}")
            if auc > best[0]:
                test_blend = (1 - w) * Tt_naji + w * tt
                best = (auc, tag, w, test_blend)

    prog(f"BEST: {best[1]} @ {best[2]:.0%} -> OOF {best[0]:.5f} | beats {BEST_KNOWN}? {best[0] > BEST_KNOWN}")
    if best[0] > BEST_KNOWN and best[3] is not None:
        out = OUT / f'submission_najiboost_{best[1]}_{int(best[2]*100)}pct_{best[0]:.5f}.csv'
        pd.DataFrame({'id': sub_ids,
                      'addicted_label': np.clip(np.asarray(best[3]), 0, 1)}).to_csv(out, index=False)
        prog(f"SAVED {out.name}")
    else:
        prog("no submission written — no gain over naji18 alone")


if __name__ == '__main__':
    main()
