# Diversity Research & Plan — Improving pool141 (Best LB 0.97105)

**Created:** 2026-08-19
**Goal:** Make the best-performing model (pool141 blend, LB 0.97105) more diverse → close the gap to top of leaderboard
**Deadline:** 2026-08-31 23:59 UTC (~12 days)
**Compute rule:** NO Kaggle GPU. GPU work → **Colab CLI**. CPU work → Kaggle CPU kernels.

---

## Part 1: Current State (verified 2026-08-19)

| Item | Status |
|---|---|
| Best LB | **0.97105** (pool141 rank-blend of public submissions, 13 Aug) |
| Current rank | **148** of 2,252 teams (was #33 on 16 Aug — others keep improving) |
| Own-model track ceiling | ~**0.9643 CV** (`xgb_opt+lgb` greedy 0.96425, stack_lgb 0.96406) |
| Own-model LB attempt | 0.95999 (17 Aug) — **invalid test**: had XGB fold-accumulation bug (now fixed) |
| Weight optimization (ens_weight kernel) | Optimized weights == uniform (0.96323 both) → **models too correlated, weights can't help** |
| Quota | 5 submissions/day, ~60 slots left until deadline |

### Kernel statuses from last session (resumed today)
| Kernel | Status | Result |
|---|---|---|
| `playground-s6e8-ensemble-weight-optimization` | COMPLETE | Flat — weight opt gave nothing |
| `playground-s6e8-pl01-pseudo-labeling` | ERROR | `train_lgb_gpu` returned 1 value, script unpacked 2 — **fixed today** |
| `playground-s6e8-pool141-improvement` | ERROR | `train_model_cv` line 182 `UnboundLocalError: params` — **fixed today**; N_SEEDS 20→5 (12h limit feasibility) |

### Bugs found & fixed today (2026-08-19)
1. `src/final_submission.py:88` — XGB test preds: `test_preds = ...` then `+= test_preds / N_FOLDS` → each fold OVERWROTE and returned `fold5 × 1.2`. Fixed to `test_preds += model.predict(...) / N_FOLDS`. **The 17 Aug LB score (0.95999) was produced by this bug — own track's true LB is unknown, likely better.**
2. `src/experiment_pl01.py` — 3 bugs: (a) `train_lgb_gpu` never returned `test_preds`, (b) XGB same overwrite bug, (c) `pseudo_labeling_round` inconsistent returns (4 vs 5 values) + rounds restarted from original data instead of accumulated. All fixed.
3. `src/experiment_pool141_improve.py` — `train_model_cv` used `(params or {})` in lgb/hgb/cb branches but `params` was never defined (only xgb branch defined it locally). Added `params=None` to signature. Also all branches now reuse `split_iter`.

---

## Part 2: Research Findings

### F1. Rank averaging > probability averaging for AUC (STRONG consensus)
- AUC depends only on **ordering** of predictions, not their values
- Rank-transforming each model's predictions puts them on a common scale → each model contributes equally, immune to calibration mismatch between models
- Sources: [Playground S6E3 discussion](https://www.kaggle.com/competitions/playground-series-s6e3/discussion/679947) ("Rank averaging is your safest default… almost always matches or beats simple mean"), [S5E5 5th place — 99-model logit stack](https://www.kaggle.com/competitions/playground-series-s5e5/writeups/5th-place-solution-a-99-model-logit-stack) ("Averaging ranks puts models on a common ordering scale — exactly what AUC rewards"), [S5E5 8th place L5 ensemble](https://www.kaggle.com/competitions/playground-series-s5e5/writeups/l5-ensemble)
- **Implication for us:** all pool blending should happen in rank space (current pool141 already does this — keep it); own-model stacking should ALSO rank-transform before blending

### F2. Hill Climbing = the proven ensemble-selection method on Kaggle
- Greedy iterative selection: start empty, each step add the model (+weight) that most improves validation AUC; allow **replacement** (same model added multiple times = bagging effect) and optionally **negative weights**
- Works on OOF predictions → leakage-safe with proper CV; handles hundreds of candidate models
- 1st-place proof points: [Chris Deotte 1st place S5E5 — GPU Hill Climbing](https://www.kaggle.com/competitions/playground-series-s5e5/writeups/chris-deotte-1st-place-gpu-hill-climbing), [1st place S5E12 — Hill Climbing + Ridge](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl) (HC-selected models → Ridge stack on rank-transformed preds), [10th place S5E6 — HC over 350 OOFs](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/ole-jakob-10th-place-solution-350-oofs-9-hillclimb)
- Academic foundation: [Caruana et al., "Getting the Most Out of Ensemble Selection"](http://www.niculescu-mizil.org/papers/enssel_most.pdf) — "overproduce and choose"
- Known limitation: greedy can miss global optimum → S5E6 writeup suggests tree search (alpha-beta) as upgrade
- **Implication for us:** replace uniform/greedy-by-CV blending of OWN models with proper rank-space hill climbing with replacement; this also naturally down-weights correlated members

### F3. Diversity comes from MODEL FAMILY, not seeds/params (for our situation)
- Our ens_weight experiment proved it empirically: LGB/XGB/HGB/CB all converge to equal weights because they're all GBDTs learning the same signal
- Own track has ZERO non-GBDT members right now (MD-04 NN, MD-05 RF — planned but never run)
- Adding 1 well-trained MLP to a GBDT ensemble typically adds more than 5 more GBDT variants — different inductive bias → decorrelated errors → blend gain
- Extra diversity levers (cheaper, smaller gains): multi-seed bagging, different CV structures per member (Stratified vs GroupKFold-age), random feature subsets per member, `dart` boosting mode
- **Implication:** highest-value new member = NN; second = ExtraTrees/RF; third = EBM

### F4. Pool-of-submissions ensembling (our best track) — how the top does it
- The S5E5/S5E6 top solutions literally built pools of 100–350 OOF/submission sets and hill-climbed them
- Our pool141 is a rank-blend of public submissions. Its weakness: members are highly correlated (public notebooks mostly copy each other) → blend saturates around 0.9710
- Two ways to push it:
  - **Refresh + prune pool**: add newer public submissions (2 weeks of new notebooks since 13 Aug), drop near-duplicates (pairwise rank-corr > 0.999 clusters → keep 1 per cluster)
  - **Inject genuinely diverse members**: our own track's output (GBDT ensemble + later NN) as an extra pool member with a small weight — this is the single most promising move per F3

### F5. LB probing as a legitimate tool (we have no OOF for pool members)
- Pool members are TEST predictions — we cannot CV them locally. The top teams' equivalent (OOF hill climb) is unavailable for the pool part
- Practical substitute: probe 1–2 blend configs per day on the public LB (5/day quota, keep ≥2/day reserve for final week). Each probe should test ONE change
- Reference: [Hill Climbing the Test Set](https://www.machinelearningmastery.com/hill-climb-the-test-set-for-machine-learning/) (careful: overfitting to public LB — private is what counts; keep probes hypothesis-driven, not exhaustive)

---

## Part 3: The Plan

### Phase 0 — Fixed & re-run (TODAY, Kaggle CPU) ✅ partially done
- [x] Fix `final_submission.py` XGB bug
- [x] Fix `experiment_pl01.py` (3 bugs)
- [x] Fix `experiment_pool141_improve.py` (params bug, seeds 20→5, split_iter)
- [ ] Re-push pl01 + pool141_improve kernels (CPU) → results decide Phase 2 members
- [ ] After they complete: re-submit corrected own-ensemble to LB (1 slot) → learn own track's TRUE LB

### Phase 1 — Pool upgrade (Kaggle CPU + LB probes) — HIGHEST VALUE
| ID | What | How | Expected |
|---|---|---|---|
| DV-01 | Refresh pool | Pull latest public submission files / notebook outputs (2 weeks stale) → pool141 → pool15x+ | +0.0000–0.0002 |
| DV-02 | Prune duplicates | Pairwise rank-corr matrix on test preds; cluster >0.999; keep best-LB member per cluster | cleaner blend, +0.0000–0.0002 |
| DV-03 | Aggregator test | On pruned pool: uniform rank-avg (current) vs AUC-weighted rank-avg vs logit-mean vs trimmed mean — 1 LB probe each for the top-2 CV-less candidates | +0.0001–0.0003 |
| DV-04 | **Inject own model** | Add fixed own-ensemble (Phase 0 output) as pool member, weights {0.02, 0.05, 0.10} — LB probe best 2 | **+0.0001–0.0005 (main lever)** |

### Phase 2 — Own-track family diversity (feeds DV-04)
| ID | What | Compute | Expected |
|---|---|---|---|
| DV-05 | MLP (3 layers, standardized + cat embeddings, 5-seed bagged) | **Colab GPU** (per rule) | diverse member, own OOF +0.0003–0.001 |
| DV-06 | ExtraTrees + RandomForest (MD-05) | Kaggle CPU | small but decorrelated |
| DV-07 | LightGBM `dart` mode | Kaggle CPU | decorrelated GBDT variant |
| DV-08 | EBM (`interpret` package) | Kaggle CPU | different bias entirely |
| DV-09 | Multi-seed bagging (fixed pool141_improve kernel) | Kaggle CPU (re-pushed) | stability + small gain |
| DV-10 | Feature-subspace ensemble (10 × random 60% features) | Kaggle CPU | small |

### Phase 3 — Rank-space hill climbing + final blend
| ID | What | Compute |
|---|---|---|
| DV-11 | Caruana HC with replacement (+negative weights) on rank-transformed OOFs of ALL own members (GBDT×4 + NN + ET + EBM + dart) | Kaggle CPU |
| DV-12 | Ridge stack on ranks (S5E12 1st-place pattern) as alternative aggregator | Kaggle CPU |
| DV-13 | Final: DV-11 output injected into pool blend (DV-04 refresh) → last LB probes; select 2 final submissions by public LB but sanity-check vs CV | LB |

### Budget & rules
- Submissions: ~5/day; reserve ≥2/day after 29 Aug; total ~40 usable probes — spend 1–2/day max, one change per probe
- Kaggle CPU kernels only; every GPU item → Colab CLI (`colab new -s s6e8-dvNN --gpu T4 …` per REMOTE_EXECUTION.md)
- Log every result in `logs/experiments.csv` (exp_id prefix `DV-xx`)
- Final-submission safety: public LB is only 20% — prefer blends that are ALSO better/CV-consistent, not just LB-lucky

### Success criteria
| Milestone | Public AUC |
|---|---|
| Current | 0.97105 (#148) |
| +Phase 1 | ≥ 0.9712 |
| +Phases 2–3 | ≥ 0.9715 (target) |

---

## Part 4: Research Sources
1. [Playground S6E3 discussion — rank averaging](https://www.kaggle.com/competitions/playground-series-s6e3/discussion/679947)
2. [S5E5 5th place — 99-model logit stack](https://www.kaggle.com/competitions/playground-series-s5e5/writeups/5th-place-solution-a-99-model-logit-stack)
3. [S5E5 8th place — L5 ensemble](https://www.kaggle.com/competitions/playground-series-s5e5/writeups/l5-ensemble)
4. [Chris Deotte 1st place S5E5 — GPU hill climbing](https://www.kaggle.com/competitions/playground-series-s5e5/writeups/chris-deotte-1st-place-gpu-hill-climbing)
5. [1st place S5E12 — HC + Ridge on ranks](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl)
6. [10th place S5E6 — HC over 350 OOFs](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/ole-jakob-10th-place-solution-350-oofs-9-hillclimb)
7. [Hill climbing discussion S3E3](https://www.kaggle.com/competitions/playground-series-s3e3/discussion/379690)
8. [Fast GPU hill climbing S4E11](https://www.kaggle.com/competitions/playground-series-s4e11/discussion/546910)
9. [Caruana et al. — Ensemble Selection paper](http://www.niculescu-mizil.org/papers/enssel_most.pdf)
10. [Guide to ensembling methods (Kaggle notebook)](https://www.kaggle.com/code/amrmahmoud123/1-guide-to-ensembling-methods)
