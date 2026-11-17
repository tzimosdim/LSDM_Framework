# LSDM Framework for Requirements Prioritization

## Overview
This repository contains the implementation of a Large-Scale Decision Making (LSDM) framework designed for Software Requirements Prioritization (RP). The framework addresses the challenges of scalability, data sparsity, and stakeholder uncertainty in large software projects. By integrating Intuitionistic Fuzzy Sets (IFSs), Recommender Systems (RS), and Shared Nearest Neighbor (SNN) clustering, the system systematically models stakeholder hesitation, reduces information overload, and guides participants through a structured Consensus Reaching Process (CRP).

## Repository Structure
The repository is organized into core source code modules, empirical datasets, evaluation notebooks, a practical practitioner evaluation protocol, and exported results directories:

### Core Modules (`src/`)
- **`data_loader.py`**: Initializes the problem space, loads the empirical software project datasets (RALIC, Pharmacy Information System, Institute Examination System [IES], and MS Word), and generates artificial multi-criteria datasets for scalability evaluation.
- **`fuzzification.py`**: Transforms continuous numeric ratings into Intuitionistic Fuzzy Values (IFVs) using linear, r-trapezoidal, or sigmoidal functions to explicitly model membership, non-membership, and hesitation.
- **`similarity.py`**: Computes pairwise stakeholder similarity matrices using the H-max distance, which accounts for the cross-evaluation of fuzzy parameters.
- **`recommender.py`**: Implements global and local collaborative filtering to predict missing evaluations, thereby mitigating data sparsity.
- **`clustering.py`**: Applies Shared Nearest Neighbor (SNN) clustering to adapt to local density variations and classify stakeholders into core (majority) and isolated (outlier) preference groups.
- **`feedback.py`**: Synthesizes Group Recommendations (GR) to guide preference modification and computes the global Consensus Degree alongside the final requirements classification.
- **`baselines.py`**: Implements comparative rating prediction baselines (User Mean, Item Mean, Crisp Cosine CF, Crisp Euclidean CF, and SVD Matrix Factorization), supervised pairwise learning-to-rank (Bayesian Personalized Ranking, BPR), and ranking-aware metrics (Normalized Discounted Cumulative Gain, nDCG@$k$).
- **`visualization.py`**: Generates publication-ready visualizations, including similarity heatmaps and Multidimensional Scaling (MDS) spatial preference maps with formal legends and core consensus boundaries.

### Datasets (`data/`)
- **`RateP-Obj.txt`, `RateP-Req.txt`, `RateP-SReq.txt`, `ralic_ranking.csv`**: The real-world RALIC benchmark dataset across three hierarchy levels (Project Objectives, Requirements, and Specific Requirements) and ground-truth rankings.
- **`pharmacy_ratings.csv`, `pharmacy_reference_priorities.csv`**: The Pharmacy Information System dataset ($42$ stakeholders evaluating $101$ requirements) and reference priority classifications.
- **`ies_multicriteria_ratings.csv`**: The multi-criteria Institute Examination System (IES) dataset ($5$ decision makers evaluating $192$ functional requirements across $3$ non-functional criteria).
- **`msword_multicriteria_ratings.csv`**: The weighted multi-criteria MS Word release-planning benchmark ($4$ stakeholders evaluating $50$ requirements across $2$ criteria).

### Experimental Notebooks (`notebooks/`)
- **`Running_Example.ipynb`**: An end-to-end demonstration of the 5-phase framework using a simulated multi-criteria decision-making scenario.
- **`RQ1_Prediction_Accuracy.ipynb`**: Evaluates the predictive accuracy (MAE) of the collaborative filtering recommender system using 10-fold cross-validation across all empirical datasets and baseline models, alongside statistical variance analysis (ANOVA with Partial Eta-Squared $\eta_p^2$ effect sizes & Tukey HSD).
- **`RQ2_Consensus_Improvement.ipynb`**: Analyzes the effectiveness of the SNN clustering (via Silhouette Coefficient) and the framework's ability to improve the global Consensus Degree ($\Delta CD$) through targeted feedback on the RALIC and Pharmacy datasets.
- **`RQ3_Ranking_Accuracy.ipynb`**: Assesses the final classification and ranking accuracy of the framework using Pearson’s correlation ($\rho$) and Normalized Discounted Cumulative Gain (nDCG@3, nDCG@5, nDCG@10, and Full nDCG) against ground-truth priorities and comparative ranking baselines (BPR, SVD, Item Mean, and zero-hesitation variants).
- **`RQ4_Scalability_Analysis.ipynb`**: Provides an empirical evaluation of the framework's computational runtime and memory consumption across expanding horizontal and vertical multi-criteria data scales.

### Experiment results (`results/`)

### Practical Evaluation Protocol
- **`Evaluation_Protocol.md`**: A structured step-by-step protocol and post-session practitioner questionnaire for applying and evaluating the 5-phase LSDM-IFS-RS framework with software development teams.

## Installation

1. Clone the repository to your local machine.
2. Ensure you have Python 3.8 or higher installed.
3. Install the required dependencies using `pip`:
```bash
pip install -r requirements.txt
```