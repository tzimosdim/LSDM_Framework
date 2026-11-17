# Practical Protocol for Applying and Evaluating the LSDM-IFS-RS Framework

## 1. Purpose and Scope
This protocol provides a step-by-step guide for software engineering teams and researchers to apply and evaluate the five-phase Large-Scale Decision-Making (LSDM) Requirements Prioritization (RP) framework in a real project environment. It is designed to assess both the quantitative performance of the process (recommendation acceptance and consensus improvement) and practitioners' perceptions (ease of use, decision transparency, and satisfaction with the final requirements ranking) [56], [63].

## 2. Participants and Roles
A prioritization session typically involves a project team of 5 to 20 stakeholders with diverse roles, such as Product Owners, Software Developers, Requirements/Business Analysts, Software Architects, and Project Managers [56]. One participant (e.g., the Product Owner or Requirements Analyst) acts as the Moderator to configure Phase 1 and coordinate the feedback rounds.

## 3. Step-by-Step Application Procedure

### Step 1: RP Problem Definition & Reference Priority Matrix (Phase 1)
1. **Define Candidate Requirements and Criteria:** The Moderator lists the $M$ candidate requirements ($F = \{f_1, \dots, f_M\}$) to be prioritized and selects $N$ evaluation criteria ($C = \{c_1, \dots, c_N\}$, e.g., Business Value, Implementation Risk, and Cost).
2. **Define Priority Classes:** The team defines $L$ target priority classes ($P = \{p_1, \dots, p_L\}$), such as upcoming software releases (`Release1`–`Release4`) or MoSCoW classes (`Must Have`, `Should Have`, `Could Have`, `Won't Have`).
3. **Set the Reference Priority Matrix:** Project experts define the numeric Reference Priority Matrix ($MCP_{\text{num}}$), specifying the ideal rating value (from 1 to 5) for each criterion in each priority class, and set the consensus threshold $\mu$ (e.g., $\mu = 0.75$) and maximum feedback rounds ($\text{Max}_{\text{rounds}} = 2$).

### Step 2: Requirements Ratings & Recommendations (Phase 2)
1. **Initial Sparse Ratings:** Each stakeholder $s_i$ independently evaluates the subset of candidate requirements they are familiar with using a 1 to 5 ordinal scale under each criterion. Unrated items are left as Not Available (`NA`), reducing information overload.
2. **Uncertainty Modeling (IFVs):** Stakeholders select their overall confidence level in their assessments (High: $\pi = 0.10$, Medium: $\pi = 0.20$, Low: $\pi = 0.30$), and the system converts the numeric ratings into Intuitionistic Fuzzy Values (IFVs) using the selected fuzzification function (e.g., linear or r-trapezoidal).
3. **Collaborative Filtering Recommendations:** The system computes pairwise H-max similarities between stakeholders, predicts the missing (`NA`) ratings using the local $k$-nearest neighbor strategy, and displays the recommended numeric ratings (1 to 5) to each stakeholder. Stakeholders may accept, adjust, or ignore the suggested ratings for their unrated items.

### Step 3: Stakeholders Clustering (Phase 3)
The system constructs the Shared Nearest Neighbor (SNN) similarity network from the stakeholders' pairwise H-max similarities, calculates the SNN density for each participant, and classifies participants into two groups:
* **Core Stakeholders:** Participants with SNN density above the threshold, representing the majority view of the project team.
* **Isolated Stakeholders:** Participants with low SNN density, representing preference outliers whose divergent ratings reduce group consensus.

### Step 4: Feedback and Preference Modification (Phase 4)
1. **Consensus Check:** The system calculates the initial overall Consensus Degree ($CD_0$). If $CD_0 \ge \mu$, the process proceeds directly to Step 5.
2. **Group Recommendation (GR) Feedback:** If $CD_0 < \mu$, the system aggregates the IFV profiles of the core stakeholders to generate the numeric Group Recommendation (GR) and displays the 2D Stakeholder Preference Map (MDS projection).
3. **Voluntary Preference Revision:** Isolated stakeholders review the Group Recommendation (GR) and the preference map, and are invited to voluntarily revise their ratings where they agree with the group perspective, or retain their original ratings where they hold critical domain knowledge. This feedback loop repeats for at most $\text{Max}_{\text{rounds}}$ iterations.

### Step 5: Consensus Measurement & Final Requirements Ranking (Phase 5)
Once the consensus threshold is reached (or $\text{Max}_{\text{rounds}}$ is completed), the system calculates the final Consensus Degree ($CD_{\text{final}}$), computes the degree of relation ($d(f_m, p_l)$) between the updated stakeholder ratings and the IFV Reference Priority Matrix, and outputs the final prioritized requirements list and release assignments.

## 4. Quantitative Process Measures
During the session, the framework automatically logs four objective metrics:
1. **Initial Sparsity (%):** Percentage of `NA` entries prior to Phase 2 recommendations.
2. **Recommendation Acceptance Rate (RAR %):** Percentage of Phase 2 recommended ratings accepted by stakeholders (within $\pm 1$ scale point).
3. **Observed Consensus Improvement ($\Delta CD$):** The increase in Consensus Degree from the initial round to the final round ($\Delta CD = CD_{\text{final}} - CD_0$).
4. **Completion Time (minutes):** Total time required by the team to reach the final prioritized list.

## 5. Post-Session Practitioner Questionnaire
After reviewing the final requirements ranking, participants complete the following short questionnaire (scored on a 5-point ordinal Likert scale: `1 = Strongly Disagree`, `2 = Disagree`, `3 = Neutral`, `4 = Agree`, `5 = Strongly Agree`) [56], [63]:

| ID | Evaluation Dimension | Statement | Scale (1–5) |
| :--- | :--- | :--- | :---: |
| **Q1** | **Ease of Use** | The 1–5 rating scale and the Reference Priority Matrix classes were easy to understand and use. | 1 – 2 – 3 – 4 – 5 |
| **Q2** | **Uncertainty Expression** | Accounting for uncertainty/hesitation ($\pi$) better reflected my confidence when evaluating unfamiliar requirements. | 1 – 2 – 3 – 4 – 5 |
| **Q3** | **Information Overload** | The Phase 2 rating recommendations helped me handle unrated requirements and reduced evaluation effort. | 1 – 2 – 3 – 4 – 5 |
| **Q4** | **Recommendation Quality** | The predicted ratings suggested by the Recommender System were accurate and aligned with my preferences. | 1 – 2 – 3 – 4 – 5 |
| **Q5** | **Decision Transparency** | The 2D Stakeholder Preference Map (MDS) and SNN clustering clearly showed the level of agreement in the team. | 1 – 2 – 3 – 4 – 5 |
| **Q6** | **Consensus Support** | The Group Recommendation (GR) provided useful feedback that helped reduce conflicting priorities among stakeholders. | 1 – 2 – 3 – 4 – 5 |
| **Q7** | **Fairness & Autonomy** | I felt free to keep my own ratings when I had strong domain reasons to disagree with the Group Recommendation. | 1 – 2 – 3 – 4 – 5 |
| **Q8** | **Final Ranking Validity** | The final prioritized requirements list and release assignments accurately reflect the needs of the project. | 1 – 2 – 3 – 4 – 5 |

**Open-Ended Feedback Questions:**
* **OQ1:** Which phase of the prioritization framework was most helpful to your decision-making, and why?
* **OQ2:** In which cases did you choose to accept or reject the Group Recommendation (GR) during the feedback phase?
* **OQ3:** What improvements would make this framework easier to adopt in your regular release planning meetings?
