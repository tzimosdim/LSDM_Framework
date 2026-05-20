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
        cbar_kws={'label': ''},
        vmin=0,
        vmax=1
    )

    # Format axes for visual clarity
    ax.tick_params(left=False, bottom=False)
    plt.title(title, fontsize=12, pad=10, fontweight='bold')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

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

    plt.figure(figsize=(8, 6))

    # Render the spatial position of individual stakeholders
    for i, label in enumerate(st_list):
        x, y = coords[i]

        # Differentiate markers based on SNN clustering classification
        if label in core_list:
            color = '#4682B4'  # SteelBlue
            size = 75  # Small circle for core

            # Plot circle but DO NOT add a text label
            plt.scatter(x, y, c=color, s=size, alpha=1.0)
            # final_label = f"{label}_core"
        else:
            color = '#E35F62'  # Reddish
            size = 150  # Medium circle for isolated
            final_label = f"{label}_isolated"

            # Plot circle AND add the text label
            plt.scatter(x, y, c=color, s=size, alpha=1.0)
            plt.text(x + 0.02, y, final_label, fontsize=9, color='black', verticalalignment='center')

    # Render the target Group Recommendation (GR) acting as the consensus focal point
    gr_x, gr_y = coords[n]
    plt.scatter(gr_x, gr_y, c='#FF9F40', s=250, alpha=1.0)
    plt.text(gr_x + 0.02, gr_y, "Group_Recommendation", fontsize=10, fontweight='bold', color='black')

    plt.title("Stakeholder Preference Map (MDS)", fontsize=14, fontweight='bold')

    # Remove quantitative axes to emphasize relative spatial grouping and distance
    plt.xticks([])
    plt.yticks([])
    plt.box(True)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()
