"""
src/data_loader.py
Phase 1: Requirements Prioritization Problem Definition & Reference Priority Matrix.
Handles the initialization of candidate requirements, stakeholders, and criteria.
Provides mechanisms to load the real-world RALIC project dataset, the Pharmacy Information
System dataset, the Institute Examination System (IES) multi-criteria dataset, the MS Word
multi-criteria dataset, or generate artificial multi-criteria datasets for scalability evaluation.
"""

import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def load_ralic(data_id=None, base_path='../data/'):
    """
    Loads the RALIC dataset representing a real-world large-scale software project.
    Extracts stakeholder ratings across three hierarchy levels (Project Objectives,
    Requirements, Specific Requirements).

    Args:
        data_id (int, optional): 1 for Objectives, 2 for Requirements, 3 for Specific Reqs.
                                 If None, concatenates all three.
        base_path (str): The relative path to the directory containing the RALIC .txt files.

    Returns:
        tuple: (df, labels, weights, knowledge_matrix)
    """
    file_map = {
        1: 'RateP-Obj.txt',
        2: 'RateP-Req.txt',
        3: 'RateP-SReq.txt'
    }

    dfs = []
    if data_id is None:
        for idx in [1, 2, 3]:
            dfs.append(pd.read_csv(os.path.join(base_path, file_map[idx]), delimiter="\t", header=None))
        df = pd.concat(dfs, axis=0)
    else:
        df = pd.read_csv(os.path.join(base_path, file_map[data_id]), delimiter="\t", header=None)

    df[2] = df[2].astype(float)

    # Format the ratings matrix: rows represent stakeholders (S) and columns represent candidate requirements (F)
    reqs = np.unique(df[1])
    cr = ['Cr1' for _ in range(len(reqs))]
    columns = pd.MultiIndex.from_tuples(list(zip(*[reqs, cr])), names=['Requirements', 'Criteria'])

    df = df.pivot_table(index=0, columns=1, values=2)
    df.columns = columns

    # Initialize uniform stakeholder weights (W = 1.0) prior to any non-cooperative behavior penalization
    labels = [0 for _ in range(df.shape[0])]
    weights = np.array([1.0 for _ in range(df.shape[0])])

    # Define the Reference Priority Matrix for the RALIC dataset (single criterion, single priority class)
    knowledge_matrix = pd.DataFrame([[5.0]], index=['Cr1'], columns=['Release1'])

    return df, labels, weights, knowledge_matrix


def load_pharmacy(base_path='../data/'):
    """
    Loads the Pharmacy Information System dataset (Gambo & Taveter, 2021).
    Contains 42 stakeholders evaluating 101 requirements on a 1-5 priority scale,
    along with the published resolved priority reference table.

    Returns:
        tuple: (df, labels, weights, knowledge_matrix, df_ground_truth)
    """
    ratings_path = os.path.join(base_path, 'pharmacy_ratings.csv')
    gt_path = os.path.join(base_path, 'pharmacy_reference_priorities.csv')

    df_raw = pd.read_csv(ratings_path, index_col=0).astype(float)
    reqs = df_raw.columns.tolist()
    cr = ['Cr1' for _ in range(len(reqs))]
    df_raw.columns = pd.MultiIndex.from_tuples(list(zip(reqs, cr)), names=['Requirements', 'Criteria'])

    labels = [0 for _ in range(df_raw.shape[0])]
    weights = np.ones(df_raw.shape[0], dtype=float)

    # Expert Reference Priority Matrix for Very High (VH = 5.0) and High (H = 4.0) classes
    knowledge_matrix = pd.DataFrame([[5.0, 4.0]], index=['Cr1'], columns=['VH', 'H'])

    df_ground_truth = pd.read_csv(gt_path)
    df_ground_truth['Requirement'] = df_ground_truth['Requirement'].astype(str).str.strip()

    return df_raw, labels, weights, knowledge_matrix, df_ground_truth


def load_ies_multicriteria(base_path='../data/'):
    """
    Loads the Institute Examination System (IES) multi-criteria dataset (Nazim et al., 2023).
    Contains 5 Decision Makers evaluating 192 Functional Requirements across 3 Non-Functional
    Criteria (NFR1: Security, NFR2: Cost, NFR3: Usability) on a 1-5 scale.

    Returns:
        tuple: (df, labels, weights, knowledge_matrix, df_ground_truth)
    """
    ies_path = os.path.join(base_path, 'ies_multicriteria_ratings.csv')
    df_long = pd.read_csv(ies_path)

    # Preserve numerical requirement ordering SR1..SR192
    req_order = [f"SR{i}" for i in range(1, 193)]
    crit_order = ['NFR1', 'NFR2', 'NFR3']
    dm_order = [f"DM{i}" for i in range(1, 6)]

    # Build wide (DM x (Requirement, Criterion)) DataFrame
    records = {}
    for dm in dm_order:
        # Filter for the specific stakeholder
        sub = df_long[df_long['Stakeholder'] == dm]

        # Pivot so that Requirements are the index and Criteria become the columns
        sub_pivot = sub.pivot(index='Requirement', columns='Criterion', values='Rating')

        row_dict = {}
        for r in req_order:
            for c in crit_order:
                # Safely extract the rating if the requirement and criterion exist
                if r in sub_pivot.index and c in sub_pivot.columns:
                    row_dict[(r, c)] = float(sub_pivot.loc[r, c])
                else:
                    row_dict[(r, c)] = np.nan
        records[dm] = row_dict

    df = pd.DataFrame.from_dict(records, orient='index')
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Requirements', 'Criteria'])

    labels = [0 for _ in range(df.shape[0])]
    weights = np.ones(df.shape[0], dtype=float)

    # Reference Priority Matrix across 3 NFR criteria (1-5 scale)
    knowledge_matrix = pd.DataFrame(
        {
            'Release1': [5.0, 5.0, 5.0],
            'Release2': [4.0, 4.0, 4.0],
            'Release3': [3.0, 3.0, 3.0],
            'Release4': [2.0, 2.0, 2.0]
        },
        index=crit_order
    )

    # Derive ground-truth reference priority from full unmasked multi-criteria consensus average
    sub_pivot = df_long.pivot_table(index='Requirement', columns='Criterion', values='Rating', aggfunc='mean')
    gt_scores = sub_pivot[crit_order].mean(axis=1).reindex(req_order)
    df_ground_truth = pd.DataFrame({
        'Requirement': req_order,
        'Hierarchy_Level': 1,
        'Ground_Truth_Score': gt_scores.values,
        'Ground_Truth_Rank': gt_scores.rank(ascending=False, method='average').values
    })

    return df, labels, weights, knowledge_matrix, df_ground_truth


def load_msword_multicriteria(base_path='../data/'):
    """
    Loads the MS Word Processing Tool multi-criteria dataset (Karim & Ruhe, 2014).
    """
    ms_path = os.path.join(base_path, 'msword_multicriteria_ratings.csv')
    df_long = pd.read_csv(ms_path)

    # Dynamically extract labels from the actual dataset rather than hardcoding
    req_order = df_long['Requirement'].unique().tolist()
    crit_order = ['Urgency', 'Value']
    st_order = df_long['Stakeholder'].unique().tolist()

    records = {}
    st_weights = []
    for st in st_order:
        sub = df_long[df_long['Stakeholder'] == st]
        st_weights.append(float(sub['Stakeholder_Weight'].iloc[0]))

        # Pivot the data to wide format to match the script's expectations
        sub_pivot = sub.pivot_table(index='Requirement', columns='Criterion', values='Rating', aggfunc='mean')

        row_dict = {}
        for r in req_order:
            for c in crit_order:
                if r in sub_pivot.index and c in sub_pivot.columns:
                    row_dict[(r, c)] = float(sub_pivot.loc[r, c])
                else:
                    row_dict[(r, c)] = np.nan
        records[st] = row_dict

    df = pd.DataFrame.from_dict(records, orient='index')
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Requirements', 'Criteria'])

    labels = [0 for _ in range(df.shape[0])]
    weights = np.array(st_weights, dtype=float) / np.max(st_weights)

    # Reference Priority Matrix for 1-9 scale
    knowledge_matrix = pd.DataFrame(
        {
            'Release1': [9.0, 9.0],
            'Release2': [6.0, 6.0],
            'Release3': [3.0, 3.0]
        },
        index=crit_order
    )

    # Stakeholder-importance weighted multi-criteria score as ground-truth reference
    df_long['Weighted_Score'] = df_long['Stakeholder_Weight'] * 0.5 * df_long['Rating']
    gt_series = df_long.groupby('Requirement')['Weighted_Score'].sum().reindex(req_order)

    # Use the dynamically extracted group column
    req_groups = df_long.drop_duplicates('Requirement').set_index('Requirement').reindex(req_order)['Group']

    df_ground_truth = pd.DataFrame({
        'Requirement': req_order,
        'Requirement_Group': req_groups.values,
        'Ground_Truth_Score': gt_series.values,
        'Ground_Truth_Rank': gt_series.rank(ascending=False, method='average').values
    })

    return df, labels, weights, knowledge_matrix, df_ground_truth


def generate_synthetic_data(n_stakeholders=100, n_reqs=100, n_cr=5, sparsity=0.60, seed=42):
    """
    Generates an artificial multi-criteria dataset to evaluate framework scalability (RQ4)
    and sparsity tolerance. Constructs the problem space including stakeholders (I),
    requirements (M), criteria (N), and the predefined priority classes (L).

    Args:
        n_stakeholders (int): Number of stakeholders (I)
        n_reqs (int): Number of requirements (M)
        n_cr (int): Number of criteria (N)
        sparsity (float): Percentage of missing values (0.0 to 1.0)
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (df_numeric, df_true_numeric, mask_missing, knowledge_matrix)
    """
    np.random.seed(seed)

    # Define the evaluation space (Candidate Requirements x Rating Criteria)
    cols = pd.MultiIndex.from_product(
        [[f"Req{i}" for i in range(1, n_reqs + 1)], [f"Cr{i}" for i in range(1, n_cr + 1)]],
        names=['Requirements', 'Criteria']
    )

    n_cols = n_reqs * n_cr
    total_cells = n_stakeholders * n_cols

    # Generate complete ground-truth numeric ratings matrix on a 1-5 continuous scale
    data_full = np.random.randint(1, 6, size=(n_stakeholders, n_cols))
    df_true_numeric = pd.DataFrame(
        data_full,
        index=[f"St{i}" for i in range(1, n_stakeholders + 1)],
        columns=cols
    )

    # Introduce missing values to simulate data sparsity and stakeholder information overload
    n_missing = int(total_cells * sparsity)
    indices_flat = np.random.choice(total_cells, n_missing, replace=False)

    data_sparse = data_full.astype(float).flatten()
    data_sparse[indices_flat] = np.nan
    data_sparse = data_sparse.reshape(n_stakeholders, n_cols)

    mask_missing = np.isnan(data_sparse)
    df_numeric = pd.DataFrame(data_sparse, index=df_true_numeric.index, columns=cols)

    # Define the numeric Reference Priority Matrix mapping ideal criteria values to 4 prioritization classes
    cpcrm_data = {
        'Must Have': [5.0, 5.0, 5.0, 2.0, 2.0],
        'Should Have': [4.0, 4.0, 4.0, 3.0, 3.0],
        'Could Have': [3.0, 3.0, 3.0, 4.0, 4.0],
        'Wont Have': [2.0, 2.0, 2.0, 5.0, 5.0]
    }

    knowledge_matrix = pd.DataFrame(
        cpcrm_data,
        index=[f"Cr{i}" for i in range(1, n_cr + 1)]
    )

    return df_numeric, df_true_numeric, mask_missing, knowledge_matrix
