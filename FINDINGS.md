# S6E8 Campaign Findings (final, 23 Aug 2026)

Deadline: 31 Aug 23:59 UTC. Best LB: **pool141 = 0.97105** (#230/2614 at last check).
Top LB: 0.97142.

## Final picks (LOCK these 28-29 Aug)
| Pick | Public LB | Reason |
|---|---|---|
| pool141 | 0.97105 | best public, stable rank-blend |
| naji18_bands | 0.97096 | honest OOF-backed, diversity hedge for private shake-up |

## Verified core formula
```
Public LB ≈ honest OOF + 0.0011   (held across 3 independent blends)
```
- naji18: OOF 0.969856 -> LB 0.970960
- library author 54-blend: 0.96943 -> 0.97062
- tabm_seed3: 0.96867 -> 0.96967

## What worked
- **Big un-pruned uniform rank-averages** of public submissions (pool94->pool141 gained)
- **Frozen-fold stacking**: StratifiedKFold(5, seed=42) on train.csv FILE ORDER —
  the community convention; all public OOF libraries stack row-for-row on it
- **CV-first rule**: never submit without honest CV backing. Saved slots repeatedly
  (DV-11 predicted ~0.9705 < 0.97105 -> skipped; DV-01/04 probes without CV -> both failed)

## What did NOT work (all verified, don't retry)
- **Pruning the pool** to near-duplicate-free 25 members: -0.0018 (DV-01). Bigger pool wins here.
- **Injecting own models into public pool**: -0.00025 (DV-04); hill-climb never picked them
- **MLP (DV-05)**: 256-128-64 + cat embeddings + 5-seed bagged = OOF 0.94041 alone,
  zero blend contribution. GBDT-only signal in this data
- **74-model public OOF library ceiling**: nested HC = 0.96944, ridge stack 0.96707,
  uniform-all 0.96698 — all below naji18's 0.96986. Route saturated
- **Digit/quantization features (DV-12)**: improved OWN singles (xgb 0.96384->0.96582,
  lgb +0.0003) but blend stayed flat at 0.96944 — signals already inside lib top members
- **naji18 + small% digit models (DV-13)**: flat/negative at every weight
- **Hyperparameter tuning / weight optimization**: flat (ens_weight experiment proved
  weights converge to uniform — models too correlated)

## DV-14 (27 Aug): exact-value TE pipeline — hypothesis confirmed, LB gain zero
Forensic (community ablations + own-data verification) showed the comp's generator uses
exact values as lookup keys: notifications/app_opens adjacent-value rate jumps 17-20x
sampling noise; daily_screen_time first-decimal-digit swings 0.663->0.733. Own track had
missed this entirely (raw floats + median impute + max_bin=255 = 0.96425 ceiling).
- Built: stringified nested-OOF TE (10 inner folds, smoothing 10) + transductive freq +
  max_bin=2047 + decimal lattice + imputed-as-extra + CatBoost native ordered TS
- Result: LGBM 0.96763 / XGB 0.96797 / CB 0.96783 / logitstack **0.96839 OOF** (+0.0041)
- BUT: corr vs naji18 = 0.990 OOF / 0.993 test; logit-stack on naji18 = 0.96986 FLAT.
  TE signal already inside pool members. Expected LB ~0.9697 < both current picks.
- Conclusion: no probe worth a slot. Representation gap closed; pool had it already
  (that's WHY pool141 = 0.97105). Remaining gap to 0.97186 = ensemble scale + public-LB
  selection zone (>0.97110 = mostly noise per raykkretzschmar; S6E2/E6/E7 public top-10
  had ZERO private top-10 survivors). Final picks unchanged: pool141 + naji18_bands.
- Assets: models/dv14_oof.csv, models/dv14_test.csv, kernel playground-s6e8-dv14-exact-value-te-pipeline

## Rules that saved us (keep for next comp)
1. Honest CV before every submission; threshold = current best known CV
2. No heavy local runs — everything on Kaggle kernels (13GB laptop RAM)
3. GPU kernels ALWAYS `"machine_shape": "NvidiaTeslaT4"` (P100 torch cu128 crashes:
   cudaErrorNoKernelImageForDevice) + `enable_gpu: true`
4. After `kaggle kernels push` / competitions submit: NEVER wait/poll
5. Public LB is a small sample — private is what counts; probe max 1-2/day, one change each

## Key assets built
- `data/pool/oof_library/` — 74-model OOF+test library (frozen folds)
- `data/pool/golem/` — 7 honest models incl. digit-feature LGBM recipe
- Kaggle dataset `adityaraj0612/s6e8-stack-pool` (+naji18), kernels dv05/dv11/dv12/naji-boost
- Own: MLP oof/test (models/dv05_oof.csv), digit lgb/xgb oof/test (dv12 kernel output)

## Last-week plan
Roz: new public notebooks scan (react only if solo >0.9710 appears).
28-29 Aug: lock final picks. 30-31 Aug: touch nothing, keep >=2 slots/day reserve.
