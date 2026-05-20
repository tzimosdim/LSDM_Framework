"""
src/recommender.py
Phase 2: Requirements Ratings Recommendations.
Applies collaborative filtering to predict missing Intuitionistic Fuzzy Value (IFV) ratings.
Mitigates data sparsity and reduces stakeholder information overload by estimating
unrated candidate requirements based on the preferences of similar stakeholders.
"""

import numpy as np
import pandas as pd
import warnings


def predict_missing_values(df_ifv, sim_matrix, k=2, strategy='global'):
    """
    Predicts missing IFV ratings to resolve data sparsity using collaborative filtering.
    """
    if strategy == 'global':
        return _predict_global(df_ifv, sim_matrix, k)
    elif strategy == 'local':
        return _predict_local(df_ifv, sim_matrix, k)
    else:
        raise ValueError("Strategy must be either 'global' or 'local'.")


def _predict_global(df_ifv, sim_matrix, k):
    """
    Calculates missing ratings using a global top-k nearest neighbors approach.
    Applies vectorized operations across the entire similarity matrix simultaneously.
    """
    users, reqs = df_ifv.index, df_ifv.columns
    n_users, n_reqs = len(users), len(reqs)

    raw_values = df_ifv.values.flatten()
    flat_list = [x if isinstance(x, tuple) else (np.nan, np.nan, np.nan) for x in raw_values]
    data_arr = np.array(flat_list)

    mu_matrix = data_arr[:, 0].reshape(n_users, n_reqs)
    nu_matrix = data_arr[:, 1].reshape(n_users, n_reqs)

    sim_values = sim_matrix.values.copy()
    np.fill_diagonal(sim_values, -1)

    if k < n_users:
        sorted_indices = np.argsort(-sim_values, axis=1, kind='stable')
        top_k_indices = sorted_indices[:, :k]
        mask_k = np.zeros_like(sim_values, dtype=bool)
        np.put_along_axis(mask_k, top_k_indices, True, axis=1)
        sim_values[~mask_k] = 0

    sim_values[sim_values < 0] = 0
    W = sim_values

    mask = ~np.isnan(mu_matrix)
    mu_filled = np.nan_to_num(mu_matrix)
    nu_filled = np.nan_to_num(nu_matrix)

    denom = W @ mask.astype(float)
    numer_mu = W @ mu_filled
    numer_nu = W @ nu_filled

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_mu = np.divide(numer_mu, denom, out=np.zeros_like(numer_mu), where=denom != 0)
        pred_nu = np.divide(numer_nu, denom, out=np.zeros_like(numer_nu), where=denom != 0)

    pred_pi = 1.0 - pred_mu - pred_nu

    mu_flat, nu_flat, pi_flat = pred_mu.flatten(), pred_nu.flatten(), pred_pi.flatten()
    is_missing = np.isnan(data_arr[:, 0])

    data_arr[is_missing, 0] = mu_flat[is_missing]
    data_arr[is_missing, 1] = nu_flat[is_missing]
    data_arr[is_missing, 2] = pi_flat[is_missing]

    final_tuples = [tuple(x) for x in data_arr]
    res_flat = np.empty(len(final_tuples), dtype=object)
    res_flat[:] = final_tuples

    return pd.DataFrame(res_flat.reshape(n_users, n_reqs), index=users, columns=reqs)


def _predict_local(df_ifv, sim_matrix, k):
    """
    Calculates missing ratings using a local top-k collaborative filtering approach.
    For each unrated requirement, it identifies the top-k similar stakeholders
    who have explicitly rated that specific requirement.
    """
    users, reqs = df_ifv.index, df_ifv.columns
    n_users, n_reqs = len(users), len(reqs)

    raw_values = df_ifv.values.flatten()
    flat_list = [x if isinstance(x, tuple) else (np.nan, np.nan, np.nan) for x in raw_values]
    data_arr = np.array(flat_list).reshape(n_users, n_reqs, 3)

    mu_matrix = data_arr[:, :, 0]
    nu_matrix = data_arr[:, :, 1]

    pred_mu, pred_nu, pred_pi = mu_matrix.copy(), nu_matrix.copy(), data_arr[:, :, 2].copy()

    # Pre-fill missing values with maximum hesitation (μ=0.0, ν=0.0, π=1.0).
    # Ensures stability during extreme sparsity when no similar stakeholders have rated the requirement.
    missing_mask = np.isnan(mu_matrix)
    pred_mu[missing_mask] = 0.0
    pred_nu[missing_mask] = 0.0
    pred_pi[missing_mask] = 1.0

    sim_values = sim_matrix.values.copy()
    np.fill_diagonal(sim_values, -1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in range(n_reqs):
            valid_mask = ~np.isnan(mu_matrix[:, m])

            # If no stakeholder in the evaluated subset has rated this requirement,
            # retain the maximum hesitation default value.
            if not valid_mask.any():
                continue

            valid_mu = mu_matrix[valid_mask, m]
            valid_nu = nu_matrix[valid_mask, m]
            sim_to_valid = sim_values[:, valid_mask]

            for u in range(n_users):
                if not np.isnan(mu_matrix[u, m]):
                    continue

                u_sims = sim_to_valid[u]
                if len(u_sims) <= k:
                    top_indices = np.arange(len(u_sims))
                else:
                    sorted_indices = np.argsort(-u_sims, kind='stable')
                    top_indices = sorted_indices[:k]

                top_sims = u_sims[top_indices]
                sum_sims = np.sum(top_sims)

                # Predict membership and non-membership values as the similarity-weighted
                # average of the k-nearest neighbors' ratings.
                if sum_sims > 0:
                    pred_mu[u, m] = float(np.sum(top_sims * valid_mu[top_indices]) / sum_sims)
                    pred_nu[u, m] = float(np.sum(top_sims * valid_nu[top_indices]) / sum_sims)
                    pred_pi[u, m] = 1.0 - pred_mu[u, m] - pred_nu[u, m]

    final_tuples = []
    for u in range(n_users):
        row = [(pred_mu[u, m], pred_nu[u, m], pred_pi[u, m]) for m in range(n_reqs)]
        final_tuples.append(row)

    return pd.DataFrame(final_tuples, index=users, columns=reqs)
