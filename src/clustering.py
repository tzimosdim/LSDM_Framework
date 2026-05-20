"""
src/clustering.py
Phase 3: Stakeholders Clustering.
Applies the Shared Nearest Neighbor (SNN) algorithm to group stakeholders according to rating similarity.
Evaluates SNN density to identify core participants (majority view) and isolated participants
(preference outliers) to support targeted feedback for consensus formation.
"""

import pandas as pd


def get_top_k_neighbors(sim_matrix, k=3):
    """
    Identifies the k-nearest neighbors for each stakeholder based on pairwise H-max similarity.

    Args:
        sim_matrix (pd.DataFrame): Pairwise similarity matrix (H-max).
        k (int): Number of nearest neighbors to retrieve.

    Returns:
        dict: A mapping of each stakeholder to a set of their top-k neighbors.
    """
    neighbors = {}
    for st in sim_matrix.index:
        # Exclude self-similarity to extract distinct nearest neighbors
        st_sims = sim_matrix.loc[st].drop(st)
        top_k = st_sims.nlargest(k).index.tolist()
        neighbors[st] = set(top_k)
    return neighbors


def calculate_snn_matrices(neighbors_map, stakeholders, k):
    """
    Constructs the Shared Nearest Neighbor (SNN) similarity graph.
    Edge weights between two stakeholders represent the number of nearest neighbors they mutually share,
    adapting effectively to local density variations.

    Args:
        neighbors_map (dict): The dictionary of neighbor sets from get_top_k_neighbors.
        stakeholders (list/pd.Index): List of stakeholder IDs.
        k (int): The total number of neighbors used (to calculate maximum distance).

    Returns:
        tuple: (snn_sim_df, snn_dist_df)
    """
    snn_sim = pd.DataFrame(0, index=stakeholders, columns=stakeholders, dtype=int)
    snn_dist = pd.DataFrame(0, index=stakeholders, columns=stakeholders, dtype=int)

    for i in stakeholders:
        for j in stakeholders:
            if i == j:
                # Self-similarity equals maximum neighbor count; internal distance is 0
                snn_sim.loc[i, j] = k
                snn_dist.loc[i, j] = 0
                continue

            # Calculate SNN similarity as the intersection of mutually shared neighbors
            if (j in neighbors_map[i]) and (i in neighbors_map[j]):
                shared = neighbors_map[i].intersection(neighbors_map[j])
                val = len(shared)
                snn_sim.loc[i, j] = val
                snn_dist.loc[i, j] = k - val
            else:
                snn_sim.loc[i, j] = 0
                snn_dist.loc[i, j] = k

    return snn_sim, snn_dist


def classify_stakeholders(snn_sim_matrix, threshold=None, quantile=None):
    """
    Calculates the SNN density for each participant to classify them into preference groups.
    Core stakeholders represent the leading opinion within a cluster, while isolated
    stakeholders represent outliers requiring preference modification guidance.

    Args:
        snn_sim_matrix (pd.DataFrame): The Shared Nearest Neighbor similarity matrix.
        threshold (float, optional): A predefined SNN density threshold.
        quantile (float, optional): A quantile (0.0 to 1.0) to dynamically establish the threshold.

    Returns:
        tuple: (densities_series, list_of_core, list_of_isolated)
    """
    # Calculate SNN density: sum of SNN similarities for each stakeholder
    densities = snn_sim_matrix.sum(axis=1) - snn_sim_matrix.values.diagonal()

    # Establish the SNN density threshold for core vs. isolated classification
    if quantile is not None:
        actual_threshold = densities.quantile(quantile)
    elif threshold is not None:
        actual_threshold = threshold
    else:
        actual_threshold = densities.median()

    core = []
    isolated = []

    # Assign stakeholders to groups based on the derived density threshold
    for st, density in densities.items():
        if density > actual_threshold:
            core.append(st)
        else:
            isolated.append(st)

    return densities, core, isolated
