# LSDM Framework for Requirements Prioritization

## Overview
This repository contains the implementation of a Large-Scale Decision Making (LSDM) framework designed for Software Requirements Prioritization (RP). The framework addresses the challenges of scalability, data sparsity, and stakeholder uncertainty in large software projects. By integrating Intuitionistic Fuzzy Sets (IFSs), Recommender Systems (RS), and Shared Nearest Neighbor (SNN) clustering, the system systematically models stakeholder hesitation, reduces information overload, and guides participants through a structured Consensus Reaching Process (CRP).

## Repository Structure
The repository is organized into core source code modules, evaluation notebooks, and data/results directories:

### Core Modules (`src/`)
* **`data_loader.py`**: Initializes the problem space, loads real-world datasets (e.g., the RALIC project), and generates artificial multi-criteria datasets for scalability evaluation.
* **`fuzzification.py`**: Transforms continuous numeric ratings into Intuitionistic Fuzzy Values (IFVs) using linear, r-trapezoidal, or sigmoidal functions to explicitly model membership, non-membership, and hesitation.
* **`similarity.py`**: Computes pairwise stakeholder similarity matrices using the H-max distance, which accounts for the cross-evaluation of fuzzy parameters.
* **`recommender.py`**: Implements global and local collaborative filtering to predict missing evaluations, thereby mitigating data sparsity.
* **`clustering.py`**: Applies Shared Nearest Neighbor (SNN) clustering to adapt to local density variations and classify stakeholders into core (majority) and isolated (outlier) preference groups.
* **`feedback.py`**: Synthesizes Group Recommendations (GR) to guide preference modification and computes the global Consensus Degree alongside the final requirements classification.
* **`visualization.py`**: Generates publication-ready visualizations, including similarity heatmaps and Multidimensional Scaling (MDS) spatial preference maps.

### Experimental Notebooks
* **`Running_Example.ipynb`**: An end-to-end demonstration of the 5-phase framework using a simulated multi-criteria decision-making scenario.
* **`RQ1_Prediction_Accuracy.ipynb`**: Evaluates the predictive accuracy of the collaborative filtering recommender system using k-fold cross-validation and statistical variance analysis (ANOVA & Tukey HSD).
* **`RQ2_Consensus_Improvement.ipynb`**: Analyzes the effectiveness of the SNN clustering (via Silhouette Coefficient) and the framework's ability to improve the global Consensus Degree through targeted feedback.
* **`RQ3_Ranking_Accuracy.ipynb`**: Assesses the final classification and ranking accuracy of the framework using Pearson correlation against the ground-truth priorities.
* **`RQ4_Scalability_Analysis.ipynb`**: Provides an empirical evaluation of the framework's computational runtime and memory consumption across expanding horizontal and vertical data scales.

## Installation

1. Clone the repository to your local machine.
2. Ensure you have Python 3.8 or higher installed.
3. Install the required dependencies using `pip`:
```bash
pip install -r requirements.txt