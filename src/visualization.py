"""
src/visualization.py
Generates visualizations for the framework's Consensus Reaching Process (CRP)
and clustering outcomes. Includes heatmaps for pairwise similarity matrices
and Multidimensional Scaling (MDS) to project the high-dimensional stakeholder
preference space into a 2D visual map.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn.manifold import MDS
from src.similarity import calculate_h_max_distance

sns.set_theme(style="whitegrid")

def plot_heatmap(matrix, title="Similarity Matrix", filename=None):
    """
    Renders a heatmap visualization of the pairwise similarity (or distance) matrix
    to illustrate the level of consensus and agreement among stakeholders.

    Args:
        matrix (pd.DataFrame): The similarity or distance matrix to plot.
        title (str): The title of the figure.
        filename (str, optional): If provided, saves the figure to this path.
    """
    plt.figure(figsize=(8, 6))

    # Apply sequential colormap to represent similarity gradients
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt=".2g",
        cmap="Blues",
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'H-max Similarity'},
        vmin=0,
        vmax=1
    )

    # Format axes for visual clarity
    ax.tick_params(left=False, bottom=False)
    plt.title(title, fontsize=12, pad=10, fontweight='bold')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=600, bbox_inches='tight')

    plt.show()


def plot_stakeholder_map(dist_matrix, core_list, isolated_list, gr_series, df_ifv, filename=None):
    """
    Applies Multidimensional Scaling (MDS) to project the high-dimensional stakeholder
    preferences into a 2D coordinate space. Visualizes the spatial distribution of
    core stakeholders, isolated stakeholders, and the generated Group Recommendation (GR).

    Args:
        dist_matrix (pd.DataFrame): The pairwise distance matrix of stakeholders.
        core_list (list): List of stakeholder IDs identified as core.
        isolated_list (list): List of stakeholder IDs identified as isolated.
        gr_series (pd.Series): The Group Recommendation IFV vector.
        df_ifv (pd.DataFrame): The complete IFV ratings matrix.
        filename (str, optional): If provided, saves the figure to this path.
    """
    st_list = dist_matrix.index.tolist()
    n = len(st_list)

    # Augment the distance matrix to include the calculated Group Recommendation (GR) alongside stakeholders
    aug_dist = np.zeros((n + 1, n + 1))
    aug_dist[:n, :n] = dist_matrix.values

    # Compute the H-max distance between each stakeholder's IFV ratings and the GR vector
    gr_vector = gr_series.tolist()
    for i, st in enumerate(st_list):
        st_vector = df_ifv.loc[st].tolist()
        dist = calculate_h_max_distance(st_vector, gr_vector)
        aug_dist[n, i] = dist
        aug_dist[i, n] = dist

    # Apply Multidimensional Scaling (MDS) to reduce the high-dimensional preference space to 2D
    mds = MDS(n_components=2, metric='precomputed', random_state=100, n_init=4, init='random')
    coords = mds.fit_transform(aug_dist)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    core_indices = [i for i, label in enumerate(st_list) if label in core_list]
    iso_indices = [i for i, label in enumerate(st_list) if label in isolated_list]

    # Plot shaded Convex Hull around Core Stakeholders if >= 3 core points exist
    if len(core_indices) >= 3:
        core_pts = coords[core_indices]
        try:
            hull = ConvexHull(core_pts)
            hull_pts = core_pts[hull.vertices]
            ax.fill(hull_pts[:, 0], hull_pts[:, 1], color='#4682B4', alpha=0.30,
                    edgecolor='#4682B4', linestyle='--', linewidth=1.2,
                    label='Core Consensus Region')
        except Exception:
            pass

    # Plot Core Stakeholders
    if core_indices:
        core_pts = coords[core_indices]
        ax.scatter(core_pts[:, 0], core_pts[:, 1], c='#4682B4', s=90, alpha=0.9,
                   edgecolors='white', linewidth=0.7, label=f'Core Stakeholders (n={len(core_indices)})')
        # Label core points only on small running-example maps
        if n <= 10:
            for idx in core_indices:
                ax.annotate(st_list[idx], (coords[idx, 0], coords[idx, 1]),
                            xytext=(6, 4), textcoords='offset points', fontsize=9.5, color='#2b4c6f')

    # Plot Isolated Stakeholders
    if iso_indices:
        iso_pts = coords[iso_indices]
        ax.scatter(iso_pts[:, 0], iso_pts[:, 1], c='#D9534F', s=90, marker='D', alpha=0.9,
                   edgecolors='black', linewidth=0.8, label=f'Isolated Stakeholders (n={len(iso_indices)})')
        for idx in iso_indices:
            ax.annotate(st_list[idx], (coords[idx, 0], coords[idx, 1]),
                        xytext=(7, -3), textcoords='offset points',
                        fontsize=9, fontweight='medium', color='#8a1f1c')

    # Plot Group Recommendation (GR)
    gr_x, gr_y = coords[n]
    ax.scatter(gr_x, gr_y, c='#FF9F40', s=180, marker='*', edgecolors='black',
               linewidth=1.1, zorder=5, label='Group Recommendation (GR)')
    ax.annotate("Group Recommendation", (gr_x, gr_y), xytext=(8, 5),
                textcoords='offset points', fontsize=10, fontweight='bold', color='black')

    ax.set_title("Stakeholder Preference & Consensus Map (MDS Projection)",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("MDS Dimension 1 (H-max Distance Projection)", fontsize=10.5)
    ax.set_ylabel("MDS Dimension 2 (H-max Distance Projection)", fontsize=10.5)
    ax.legend(loc='best', frameon=True, facecolor='white', framealpha=0.95, fontsize=9.5)
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=600, bbox_inches='tight')

    plt.show()
