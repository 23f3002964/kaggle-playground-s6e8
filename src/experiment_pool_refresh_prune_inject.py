"""
DV-01/DV-02/DV-04: Pool141 Refresh, Prune & Inject Own-Track
1. Collect latest public submission test predictions
2. Prune near-duplicates (pairwise rank-corr > 0.999)
3. Inject own-track ensemble at weights {1%, 2%, 5%, 10%}
4. Generate submission candidates for LB probing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('../data')
SUB_DIR = Path('../submissions')
POOL_DIR = Path('../data/pool')
LOGS_DIR = Path('../logs')
LOGS_DIR.mkdir(exist_ok=True)

SEED = 42

def prog(msg):
    print(f"[POOL] {msg}", flush=True)


def load_sample_sub():
    return pd.read_csv(DATA_DIR / 'sample_submission.csv')['id'].values


def load_own_track_preds():
    """Load our fixed own-track ensemble predictions"""
    # Try multiple locations
    paths = [
        SUB_DIR / 'submission_ensemble_fixed_0.96425.csv',
        Path('/kaggle/working/submission_ensemble_fixed_0.96425.csv'),
        SUB_DIR / 'submission_ensemble_0.96425.csv',
    ]
    for p in paths:
        if p.exists():
            prog(f"Loading own-track from {p}")
            return pd.read_csv(p)['addicted_label'].values
    raise FileNotFoundError("Own-track ensemble submission not found. Run final_submission_fixed first.")


def collect_public_submissions():
    """
    Collect test predictions from public notebooks/submissions.
    In practice, you'd download from:
    - Kaggle notebook outputs (public kernels with submission files)
    - Discussion threads with submission CSVs
    - Your existing pool/ folder submissions
    """
    prog("Collecting public submissions...")
    
    # Start with existing pool submissions
    pool_subs = {}
    for p in POOL_DIR.glob('**/*.csv'):
        try:
            df = pd.read_csv(p)
            if 'addicted_label' in df.columns and len(df) == 296302:
                name = p.stem
                pool_subs[name] = df['addicted_label'].values
                prog(f"  Loaded {name}: {len(df)} rows")
        except Exception as e:
            prog(f"  Skipped {p}: {e}")
    
    # Also check submissions/ folder
    for p in SUB_DIR.glob('*.csv'):
        try:
            df = pd.read_csv(p)
            if 'addicted_label' in df.columns and len(df) == 296302:
                name = p.stem
                if name not in pool_subs:
                    pool_subs[name] = df['addicted_label'].values
                    prog(f"  Loaded {name}: {len(df)} rows")
        except Exception as e:
            prog(f"  Skipped {p}: {e}")
    
    prog(f"Total submissions collected: {len(pool_subs)}")
    return pool_subs


def rank_transform(preds):
    """Convert probabilities to normalized ranks [0, 1]"""
    return rankdata(preds) / len(preds)


def pairwise_rank_corr(subs_dict):
    """Compute pairwise Spearman correlation matrix"""
    names = list(subs_dict.keys())
    n = len(names)
    corr_matrix = np.ones((n, n))
    
    prog(f"Computing pairwise correlations for {n} submissions...")
    
    ranked = {name: rank_transform(preds) for name, preds in subs_dict.items()}
    
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i < j:
                rho, _ = spearmanr(ranked[name_i], ranked[name_j])
                corr_matrix[i, j] = corr_matrix[j, i] = rho
    
    return names, corr_matrix, ranked


def cluster_submissions(names, corr_matrix, threshold=0.999):
    """
    Cluster submissions by rank correlation > threshold.
    Keep the one with highest public LB score per cluster.
    """
    n = len(names)
    visited = [False] * n
    clusters = []
    
    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        for j in range(i+1, n):
            if not visited[j] and corr_matrix[i, j] >= threshold:
                cluster.append(j)
                visited[j] = True
        clusters.append(cluster)
    
    prog(f"Found {len(clusters)} clusters at threshold {threshold}")
    
    # For now, just report clusters (LB scores not available for all)
    # In practice, you'd keep the best LB per cluster
    for idx, cluster in enumerate(clusters):
        cluster_names = [names[i] for i in cluster]
        prog(f"  Cluster {idx+1} ({len(cluster)}): {cluster_names[:5]}{'...' if len(cluster) > 5 else ''}")
    
    return clusters


def prune_submissions(names, corr_matrix, clusters, keep_best_lb=None):
    """
    From each cluster, keep one representative.
    If LB scores available, keep best. Otherwise keep first (or most diverse).
    """
    keep_indices = []
    
    for cluster in clusters:
        if len(cluster) == 1:
            keep_indices.append(cluster[0])
        else:
            # Without LB scores, keep the one with lowest average correlation to others
            # (most diverse within cluster)
            best_idx = cluster[0]
            best_score = float('inf')
            for idx in cluster:
                avg_corr = np.mean([corr_matrix[idx, j] for j in cluster if j != idx])
                if avg_corr < best_score:
                    best_score = avg_corr
                    best_idx = idx
            keep_indices.append(best_idx)
            prog(f"  Kept {names[best_idx]} (avg corr to cluster: {best_score:.4f})")
    
    pruned_names = [names[i] for i in keep_indices]
    prog(f"Pruned from {len(names)} to {len(pruned_names)} submissions")
    return pruned_names


def blend_rank_avg(preds_dict, names, weights=None):
    """Uniform or weighted rank average"""
    if weights is None:
        weights = np.ones(len(names)) / len(names)
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()
    
    ranked = np.column_stack([rank_transform(preds_dict[n]) for n in names])
    blended = ranked @ weights
    return blended


def inject_own_track(pool_blend, own_track, weights):
    """Inject own-track into pool blend at different weights"""
    results = {}
    for w in weights:
        # Rank-space blend
        pool_rank = rank_transform(pool_blend)
        own_rank = rank_transform(own_track)
        blended = (1 - w) * pool_rank + w * own_rank
        results[f'w_{int(w*100)}pct'] = blended
    return results


def main():
    prog("="*60)
    prog("Pool141 Refresh, Prune & Inject (DV-01/DV-02/DV-04)")
    prog("="*60)
    
    # 1. Load own-track predictions
    own_track = load_own_track_preds()
    prog(f"Own-track loaded: mean={own_track.mean():.5f}, std={own_track.std():.5f}")
    
    # 2. Collect public submissions
    pool_subs = collect_public_submissions()
    
    if len(pool_subs) < 2:
        prog("ERROR: Need at least 2 public submissions for pooling")
        return
    
    # 3. Compute pairwise rank correlations
    names, corr_matrix, ranked = pairwise_rank_corr(pool_subs)
    
    # 4. Cluster at 0.999 threshold
    clusters = cluster_submissions(names, corr_matrix, threshold=0.999)
    
    # 5. Prune (keep most diverse per cluster)
    pruned_names = prune_submissions(names, corr_matrix, clusters)
    
    # 6. Uniform rank average of pruned pool
    pool_blend = blend_rank_avg(pool_subs, pruned_names)
    prog(f"Pruned pool uniform rank-avg: mean={pool_blend.mean():.5f}")
    
    # 7. Inject own-track at different weights
    injection_weights = [0.01, 0.02, 0.05, 0.10]
    injected = inject_own_track(pool_blend, own_track, injection_weights)
    
    # 8. Generate submission files
    sub_ids = load_sample_sub()
    
    prog("Generating submission candidates...")
    
    # Base pool blend
    base_path = SUB_DIR / 'submission_dv01_pruned_pool.csv'
    pd.DataFrame({'id': sub_ids, 'addicted_label': np.clip(pool_blend, 0, 1)}).to_csv(base_path, index=False)
    prog(f"  Base pool: {base_path}")
    
    # Injected variants
    for name, preds in injected.items():
        path = SUB_DIR / f'submission_dv04_inject_{name}.csv'
        pd.DataFrame({'id': sub_ids, 'addicted_label': np.clip(preds, 0, 1)}).to_csv(path, index=False)
        prog(f"  {name}: {path}")
    
    # 9. Log experiment
    exp_file = LOGS_DIR / 'experiments.csv'
    row = pd.DataFrame({
        'method': ['dv01_pruned_pool'] + [f'dv04_inject_{k}' for k in injected.keys()],
        'cv_auc': [np.nan] * (1 + len(injected)),  # No CV for pool blends
        'exp_id': ['DV-01'] + ['DV-04'] * len(injected),
        'category': ['Pool'] * (1 + len(injected)),
        'description': [
            f'Uniform rank-avg of {len(pruned_names)} pruned public submissions'
        ] + [
            f'Pruned pool + {int(w*100)}% own-track (rank-space)'
            for w in injection_weights
        ],
        'status': ['pending_LB'] * (1 + len(injected))
    })
    
    if exp_file.exists():
        pd.concat([pd.read_csv(exp_file), row], ignore_index=True).to_csv(exp_file, index=False)
    else:
        row.to_csv(exp_file, index=False)
    
    prog("Experiments logged")
    prog("="*60)
    prog("NEXT: Submit top 2 injected variants to public LB")
    prog("  Suggested: 1% and 5% (DV-13 showed 1% XGB best, higher = worse)")
    prog("="*60)


if __name__ == '__main__':
    main()