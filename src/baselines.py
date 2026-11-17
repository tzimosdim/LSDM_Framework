"""
src/baselines.py
Implements comparative baseline models and ranking-aware metrics requested during peer review:
1. Simple Heuristics: Item Mean, User Mean
2. Crisp Collaborative Filtering Ablations: Crisp Cosine k-NN, Crisp Euclidean k-NN
3. Matrix Factorization: Truncated SVD (Latent Factor Model)
4. Supervised Learning-to-Rank: Pairwise Bayesian Personalized Ranking (BPR)
5. Ranking-Aware Evaluation Metrics: nDCG@k (Normalized Discounted Cumulative Gain)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


# ==============================================================================
# PART 1: RATING PREDICTION BASELINES (RQ1)
# ==============================================================================

def predict_item_mean(df_train, default_val=3.0):
    """Predicts missing entries using the mean observed rating of each requirement/criterion."""
    global_mean = np.nanmean(df_train.values)
    if np.isnan(global_mean):
        global_mean = default_val
    col_means = df_train.mean(axis=0).fillna(global_mean)
    return df_train.fillna(col_means)


def predict_user_mean(df_train, default_val=3.0):
    """Predicts missing entries using the mean observed rating of each stakeholder."""
    global_mean = np.nanmean(df_train.values)
    if np.isnan(global_mean):
        global_mean = default_val
    row_means = df_train.mean(axis=1).fillna(global_mean)
    df_pred = df_train.T.fillna(row_means).T
    df_pred.columns = df_train.columns
    return df_pred


def get_crisp_similarity(df_train, metric='euclidean', scale_min=-1.0, scale_max=5.0):
    """
    Computes pairwise stakeholder similarity directly on crisp numeric ratings
    (without Intuitionistic Fuzzy Sets or hesitation modeling).
    """
    X = df_train.values.astype(float)
    n_users = X.shape[0]
    sim = np.eye(n_users, dtype=float)
    scale_range = max(1e-6, float(scale_max - scale_min))

    for i in range(n_users):
        for j in range(i):
            mask = (~np.isnan(X[i])) & (~np.isnan(X[j]))
            if not np.any(mask):
                s = 0.0
            else:
                vi = X[i, mask]
                vj = X[j, mask]
                if metric == 'cosine':
                    ni = np.linalg.norm(vi)
                    nj = np.linalg.norm(vj)
                    if ni > 0 and nj > 0:
                        s = float(np.dot(vi, vj) / (ni * nj))
                        s = max(0.0, (s + 1.0) / 2.0)
                    else:
                        s = 0.0
                elif metric == 'euclidean':
                    dist = np.sqrt(np.mean(((vi - vj) / scale_range) ** 2))
                    s = max(0.0, 1.0 - float(dist))
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
            sim[i, j] = s
            sim[j, i] = s

    return pd.DataFrame(sim, index=df_train.index, columns=df_train.index)


def predict_crisp_cf(df_train, k=10, metric='euclidean', strategy='local',
                     scale_min=-1.0, scale_max=5.0):
    """
    Predicts missing ratings using standard Crisp k-NN Collaborative Filtering.
    """
    sim_df = get_crisp_similarity(df_train, metric=metric, scale_min=scale_min, scale_max=scale_max)
    sim_vals = sim_df.values.copy()
    np.fill_diagonal(sim_vals, -1.0)

    X = df_train.values.astype(float)
    pred = X.copy()
    n_users, n_items = X.shape
    global_mean = np.nanmean(X)
    if np.isnan(global_mean):
        global_mean = (scale_min + scale_max) / 2.0

    if strategy == 'global':
        top_k_idx = np.argsort(-sim_vals, axis=1)[:, :min(k, n_users - 1)]
        for u in range(n_users):
            nbrs = top_k_idx[u]
            sims = np.maximum(0.0, sim_vals[u, nbrs])
            for m in range(n_items):
                if not np.isnan(X[u, m]):
                    continue
                valid = ~np.isnan(X[nbrs, m])
                if np.any(valid) and np.sum(sims[valid]) > 0:
                    pred[u, m] = np.sum(sims[valid] * X[nbrs[valid], m]) / np.sum(sims[valid])
                else:
                    pred[u, m] = global_mean
    else:  # local strategy
        for m in range(n_items):
            valid_users = np.where(~np.isnan(X[:, m]))[0]
            if len(valid_users) == 0:
                pred[:, m] = global_mean
                continue
            for u in range(n_users):
                if not np.isnan(X[u, m]):
                    continue
                u_sims = sim_vals[u, valid_users]
                order = np.argsort(-u_sims)[:min(k, len(valid_users))]
                top_sims = np.maximum(0.0, u_sims[order])
                if np.sum(top_sims) > 0:
                    pred[u, m] = np.sum(top_sims * X[valid_users[order], m]) / np.sum(top_sims)
                else:
                    pred[u, m] = np.mean(X[valid_users, m])

    return pd.DataFrame(pred, index=df_train.index, columns=df_train.columns)


def predict_svd(df_train, n_factors=5, n_iter=15, scale_min=-1.0, scale_max=5.0):
    """
    Predicts missing ratings via iterative low-rank SVD Matrix Factorization
    centered around stakeholder/item biases.
    """
    X = df_train.values.astype(float)
    mask = ~np.isnan(X)
    global_mean = np.nanmean(X) if np.any(mask) else (scale_min + scale_max) / 2.0

    # Initialize missing entries with item means
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), global_mean, col_means)
    X_filled = np.where(mask, X, col_means[np.newaxis, :])

    rank = max(1, min(n_factors, min(X.shape) - 1))
    for _ in range(n_iter):
        user_Bias = np.mean(X_filled, axis=1, keepdims=True)
        centered = X_filled - user_Bias
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)
        recon = (U[:, :rank] * s[:rank]) @ Vt[:rank, :] + user_Bias
        recon = np.clip(recon, scale_min, scale_max)
        X_filled = np.where(mask, X, recon)

    return pd.DataFrame(recon, index=df_train.index, columns=df_train.columns)


# ==============================================================================
# PART 2: SUPERVISED PAIRWISE LEARNING-TO-RANK (BPR) & RANKING METRICS (RQ3)
# ==============================================================================

def rank_bpr(df_numeric, n_factors=8, lr=0.03, reg=0.01, n_epochs=60, seed=100):
    """
    Implements supervised pairwise Bayesian Personalized Ranking (BPR) adapted to
    explicit stakeholder requirement ratings. Constructs preference pairs (i > j)
    whenever a stakeholder rates requirement i strictly higher than requirement j,
    and optimizes latent user/item factors via BPR log-sigmoid gradient ascent.

    Returns:
        pd.DataFrame: DataFrame with ['Requirement', 'Release1'] score per requirement.
    """
    rng = np.random.default_rng(seed)

    # Aggregate multi-criteria ratings per (stakeholder, requirement) if N > 1
    reqs = df_numeric.columns.get_level_values(0).unique().tolist()
    n_users = len(df_numeric.index)
    n_reqs = len(reqs)

    R = np.zeros((n_users, n_reqs), dtype=float)
    for m_idx, r in enumerate(reqs):
        sub = df_numeric[r].values.astype(float)
        R[:, m_idx] = np.nanmean(sub, axis=1)

    # Extract observed pairwise preference comparisons (u, i, j) where R[u, i] > R[u, j]
    pairs = []
    for u in range(n_users):
        rated_idx = np.where(~np.isnan(R[u]))[0]
        for a in range(len(rated_idx)):
            for b in range(len(rated_idx)):
                i, j = rated_idx[a], rated_idx[b]
                if R[u, i] > R[u, j]:
                    pairs.append((u, i, j))

    if not pairs:
        return pd.DataFrame({'Requirement': reqs, 'Release1': np.zeros(n_reqs)})

    pairs = np.array(pairs, dtype=int)
    k = min(n_factors, max(2, min(n_users, n_reqs) - 1))
    W = rng.normal(0.0, 0.1, size=(n_users, k))
    H = rng.normal(0.0, 0.1, size=(n_reqs, k))
    b_item = np.zeros(n_reqs, dtype=float)

    for _ in range(n_epochs):
        perm = rng.permutation(len(pairs))
        for idx in perm:
            u, i, j = pairs[idx]
            x_uij = (b_item[i] - b_item[j]) + np.dot(W[u], H[i] - H[j])
            sigmoid = 1.0 / (1.0 + np.exp(np.clip(x_uij, -20.0, 20.0)))

            grad = sigmoid
            b_item[i] += lr * (grad - reg * b_item[i])
            b_item[j] += lr * (-grad - reg * b_item[j])

            w_u = W[u].copy()
            h_i = H[i].copy()
            h_j = H[j].copy()

            W[u] += lr * (grad * (h_i - h_j) - reg * w_u)
            H[i] += lr * (grad * w_u - reg * h_i)
            H[j] += lr * (-grad * w_u - reg * h_j)

    pred_matrix = W @ H.T + b_item[np.newaxis, :]
    collective_scores = np.mean(pred_matrix, axis=0)

    return pd.DataFrame({'Requirement': reqs, 'Release1': collective_scores})


def rank_from_matrix(df_completed):
    """
    Aggregates a completed crisp rating matrix across stakeholders and criteria
    into a single priority score per requirement for RQ3 comparison.
    """
    reqs = df_completed.columns.get_level_values(0).unique().tolist()
    scores = []
    for r in reqs:
        sub = df_completed[r].values.astype(float)
        scores.append(float(np.nanmean(sub)))
    return pd.DataFrame({'Requirement': reqs, 'Release1': scores})


def calculate_group_ndcg(df_group, score_col, true_rank_col='Ground_Truth_Rank_Cat', k=None):
    """
    Computes Normalized Discounted Cumulative Gain (nDCG@k) for a requirement category.
    Converts ground-truth ranks (where 1 = highest priority) into non-negative
    relevance grades: rel = max_rank - true_rank + 1.
    """
    valid_df = df_group.dropna(subset=[score_col, true_rank_col])
    if len(valid_df) < 2:
        return np.nan

    true_ranks = valid_df[true_rank_col].values.astype(float)
    max_r = np.max(true_ranks)
    relevances = np.maximum(0.0, max_r - true_ranks + 1.0)

    if np.all(relevances == relevances[0]):
        return np.nan

    pred_scores = valid_df[score_col].values.astype(float)
    return float(ndcg_score([relevances], [pred_scores], k=k))
