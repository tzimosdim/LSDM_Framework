"""
src/data_loader.py
Phase 1: Requirements Prioritization Problem Definition & Reference Priority Matrix.
Handles the initialization of candidate requirements, stakeholders, and criteria.
Provides mechanisms to load the real-world RALIC project dataset or generate artificial
multi-criteria datasets for framework scalability evaluation.
"""

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
            dfs.append(pd.read_csv(f'{base_path}{file_map[idx]}', delimiter="\t", header=None))
        df = pd.concat(dfs, axis=0)
    else:
        df = pd.read_csv(f'{base_path}{file_map[data_id]}', delimiter="\t", header=None)

    df[2] = df[2].astype(float)

    # Format the ratings matrix: rows represent stakeholders (S) and columns represent candidate requirements (F)
    reqs = np.unique(df[1])
    cr = [f'Cr1' for _ in range(len(reqs))]
    columns = pd.MultiIndex.from_tuples(list(zip(*[reqs, cr])), names=['Requirements', 'Criteria'])

    df = df.pivot_table(index=0, columns=1, values=2)
    df.columns = columns

    # Initialize uniform stakeholder weights (W = 1.0) prior to any non-cooperative behavior penalization
    labels = [0 for _ in range(df.shape[0])]
    weights = np.array([1.0 for _ in range(df.shape[0])])

    # Define the Reference Priority Matrix for the RALIC dataset (single criterion, single priority class)
    knowledge_matrix = pd.DataFrame([[5.0]], index=['Cr1'], columns=['Release1'])

    return df, labels, weights, knowledge_matrix


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
