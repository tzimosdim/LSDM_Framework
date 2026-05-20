"""
src/feedback.py
Handles Phase 4: Feedback and Preference Modification, and
Phase 5: Consensus Measurement and Final Requirements Ranking.
"""


import numpy as np
import pandas as pd


def calculate_group_recommendation(complete_ifv_df, core_stakeholders):
    """
    Phase 4: Calculates the Group Recommendation (GR) by aggregating the
    predicted IFV ratings of the identified core stakeholders.
    This recommendation is used to guide isolated stakeholders toward the
    common decision, thereby improving the overall consensus level.

    Args:
        complete_ifv_df (pd.DataFrame): The fully predicted IFV matrix.
        core_stakeholders (list): List of stakeholder IDs identified as core.

    Returns:
        pd.Series: A series containing the GR IFV tuple for each requirement/criterion.
    """
    core_df = complete_ifv_df.loc[core_stakeholders]
    gr_ifv = {}

    for col in core_df.columns:
        ratings = core_df[col].tolist()

        # Extract membership (μ) and non-membership (ν) values, excluding missing data
        mus = [r[0] for r in ratings if not np.isnan(r[0])]
        nus = [r[1] for r in ratings if not np.isnan(r[1])]

        if mus:
            avg_mu = sum(mus) / len(mus)
            avg_nu = sum(nus) / len(nus)
            gr_ifv[col] = (avg_mu, avg_nu, 1.0 - avg_mu - avg_nu)
        else:
            gr_ifv[col] = (np.nan, np.nan, np.nan)

    return pd.Series(gr_ifv)


def calculate_consensus_degree(sim_matrix):
    """
    Phase 5a: Evaluates the global Consensus Degree (CD) among stakeholders
    based on their pairwise similarities. This determines whether the
    consensus threshold is reached or if further preference modification is required.

    Args:
        sim_matrix (pd.DataFrame): The pairwise similarity matrix.

    Returns:
        float: The consensus degree [0.0, 1.0].
    """
    n = sim_matrix.shape[0]

    # Calculate the sum of pairwise similarities, excluding self-evaluations
    sum_sim = sim_matrix.sum().sum() - n
    total_pairs = n * (n - 1)

    if total_pairs == 0:
        return 0.0

    return round(sum_sim / total_pairs, 4)


def calculate_final_classification(df_updated, df_expert_ifv):
    """
    Phase 5b: Computes the Degree of Relation between the stakeholders' final ratings
    and the expert-defined Reference Priority Matrix. Assigns each candidate requirement
    to the prioritization class with the highest relation score.

    Employs vectorized tensor operations to support extreme scalability (RQ4)
    for large-scale requirement sets.

    Args:
        df_updated (pd.DataFrame): The updated stakeholder ratings after GR feedback.
        df_expert_ifv (pd.DataFrame): The IFV Reference Priority Matrix.

    Returns:
        pd.DataFrame: A mapping of each requirement to its final priority class/release.
    """
    stakeholders = df_updated.index

    # Extract unique requirements and criteria while preserving original hierarchy order
    requirements = df_updated.columns.get_level_values(0).unique()
    criteria = df_updated.columns.get_level_values(1).unique()
    releases = df_expert_ifv.columns

    I, M, C, P = len(stakeholders), len(requirements), len(criteria), len(releases)

    # Construct a 4D tensor representing Stakeholders, Requirements, Criteria, and IFV components
    raw_vals = df_updated.values.flatten()
    flat_list = [x if isinstance(x, tuple) else (np.nan, np.nan, np.nan) for x in raw_vals]
    st_tensor = np.array(flat_list, dtype=float).reshape(I, M, C, 3)

    # Construct a 3D tensor for the Reference Priority Matrix (Priority Classes, Criteria, IFV)
    expert_tensor = np.zeros((P, C, 3))
    for p_idx, release in enumerate(releases):
        for c_idx, crit in enumerate(criteria):
            val = df_expert_ifv.loc[crit, release]
            if isinstance(val, tuple):
                expert_tensor[p_idx, c_idx, :] = val
            else:
                expert_tensor[p_idx, c_idx, :] = [np.nan, np.nan, np.nan]

    # Align tensor dimensions for vectorized mathematical broadcasting
    st_b = st_tensor[:, :, np.newaxis, :, :]  # Shape: (I, M, 1, C, 3)
    exp_b = expert_tensor[np.newaxis, np.newaxis, :, :, :]  # Shape: (1, 1, P, C, 3)

    # Compute the H-max distance, incorporating cross-evaluation between membership and non-membership
    term3 = np.abs(np.maximum(st_b[..., 0], exp_b[..., 1]) - np.maximum(st_b[..., 1], exp_b[..., 0]))
    dist_components = np.abs(st_b[..., 0] - exp_b[..., 0]) + np.abs(st_b[..., 1] - exp_b[..., 1]) + term3

    # Calculate the valid distance strictly across commonly rated criteria to handle data sparsity
    valid_mask = ~np.isnan(dist_components)
    count_c = np.sum(valid_mask, axis=3)  # Counts valid criteria. Shape: (I, M, P)
    sum_dist_c = np.nansum(dist_components, axis=3)  # Safely sums ignoring NaNs

    # Average the H-max distance across all valid criteria
    with np.errstate(divide='ignore', invalid='ignore'):
        h_max_dist = sum_dist_c / (3 * count_c)

    # Apply maximum distance (1.0) for completely unrated requirements
    h_max_dist = np.where(count_c == 0, 1.0, h_max_dist)

    # Calculate the Degree of Relation as the complement of the H-max distance averaged across stakeholders
    similarity = 1.0 - h_max_dist
    degree_of_relation = np.mean(similarity, axis=0)  # Shape: (M, P)

    # Determine the final prioritization class based on the highest Degree of Relation
    best_release_indices = np.argmax(degree_of_relation, axis=1)

    results = []
    for m in range(M):
        row = {
            'Requirement': requirements[m],
            'Final_Release': releases[best_release_indices[m]]
        }
        # Record the specific degree of relation to each prioritization class
        for p_idx, release in enumerate(releases):
            row[release] = round(degree_of_relation[m, p_idx], 6)

        results.append(row)

    return pd.DataFrame(results)


def calculate_classification(df_final_ifv, df_cpcrm_ifv):
    """
    Phase 5B: Classifies requirements into software releases.
    Calculates the Degree of Relation between the group's final ratings
    and the expert Reference Priority Matrix using the H-max distance.

    Args:
        df_final_ifv (pd.DataFrame): The final IFV ratings matrix after feedback.
        df_cpcrm_ifv (pd.DataFrame): The Expert Reference Priority Matrix (IFVs).

    Returns:
        pd.DataFrame: A matrix of Degrees of Relation for each requirement against each release.
    """
    from src.similarity import calculate_h_max_distance

    # Extract candidate requirements and target priority classes
    reqs = df_final_ifv.columns.get_level_values(0).unique()
    releases = df_cpcrm_ifv.columns

    # Initialize the relation degree matrix
    classification_df = pd.DataFrame(index=reqs, columns=releases, dtype=float)

    for req in reqs:
        for release in releases:
            dist_sum = 0
            count = 0

            # Evaluate the distance of stakeholder assessments against the expert benchmark
            for st in df_final_ifv.index:
                # Extract the criteria IFVs for this specific requirement
                st_vector = df_final_ifv.loc[st, req].tolist()

                # Extract the criteria IFVs for this release target
                expert_vector = df_cpcrm_ifv[release].tolist()

                # Calculate H-max distance
                dist = calculate_h_max_distance(st_vector, expert_vector)

                if not np.isnan(dist):
                    dist_sum += dist
                    count += 1

            # Average distance across all stakeholders for this requirement-to-release match
            avg_dist = dist_sum / count if count > 0 else 1.0

            # The Degree of Relation is the complement of the H-max distance
            classification_df.loc[req, release] = 1.0 - avg_dist

    return classification_df