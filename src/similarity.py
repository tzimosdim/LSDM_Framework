"""
src/similarity.py
Calculates pairwise distance and similarity matrices between stakeholders.
Employs the H-max distance measure, which incorporates cross-evaluation
between membership and non-membership values, providing a comprehensive
comparison of Intuitionistic Fuzzy Set (IFS) assessments.
"""

import numpy as np
import pandas as pd


def calculate_h_max_distance(ifv_list_1, ifv_list_2):
    """
    Calculates the H-max distance between two stakeholder IFV arrays.
    Considers the cross-evaluation between membership and non-membership values.
    To handle data sparsity, the calculation is restricted to requirements
    that both stakeholders have mutually rated.
    """
    dist_sum = 0
    count = 0

    for val1, val2 in zip(ifv_list_1, ifv_list_2):
        # Skip if either value is not an IFV tuple or contains missing data (NaN)
        if not isinstance(val1, tuple) or not isinstance(val2, tuple): continue
        if np.isnan(val1[0]) or np.isnan(val2[0]): continue

        mu1, nu1, _ = val1
        mu2, nu2, _ = val2

        # Calculate H-max distance components including cross-evaluation
        term1 = abs(mu1 - mu2)
        term2 = abs(nu1 - nu2)
        term3 = abs(max(mu1, nu2) - max(mu2, nu1))

        dist_sum += (term1 + term2 + term3)
        count += 1

    # Return maximum distance (1.0) if stakeholders share no common ratings
    if count == 0:
        return 1.0

    return (1 / (3 * count)) * dist_sum


def get_similarity_matrices(df_ifv):
    """
    Constructs the stakeholders' pairwise distance matrix (DM_SS) and
    similarity matrix (SM_SS) based on the H-max measure.
    Optimized by calculating the lower triangular matrix and mirroring the results.
    """
    stakeholders = df_ifv.index
    n = len(stakeholders)

    # Initialize symmetric distance and similarity matrices
    sim_df = pd.DataFrame(index=stakeholders, columns=stakeholders, dtype=float)
    dist_df = pd.DataFrame(index=stakeholders, columns=stakeholders, dtype=float)

    # Convert DataFrame to list of lists for faster row iteration
    rows = [df_ifv.iloc[i].tolist() for i in range(n)]

    for i in range(n):
        # Set self-similarity diagonal to maximum (1.0) and distance to minimum (0.0)
        sim_df.iloc[i, i] = 1.0
        dist_df.iloc[i, i] = 0.0

        # Calculate lower triangle pairwise distances using only commonly rated values
        for j in range(i):
            dist = calculate_h_max_distance(rows[i], rows[j])

            # Mirror computed values to the upper triangle to complete the symmetric matrices
            dist_df.iloc[i, j] = dist
            dist_df.iloc[j, i] = dist

            # Calculate pairwise similarity as the complement of the H-max distance
            sim_val = 1 - dist
            sim_df.iloc[i, j] = sim_val
            sim_df.iloc[j, i] = sim_val

    return dist_df, sim_df
