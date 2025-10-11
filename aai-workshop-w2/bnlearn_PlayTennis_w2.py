#############################################################################
# bnlearn_PlayTennis2.py
#
# Implements the Naive Bayes classifier for simple probabilistic inference.
# It makes use of the bnlearn library for parameter estimation and inference.
# It also makes use of the pmgpy library for the same purpose but using a
# more advanced parameter learning technique (MLE with Dirichlet priors).
#
# In total, this program has four ways of learning parameters:
# 1. MLE without smoothing
# 2. MLE with Laplace smoothing
# 3. MLE with Additive smoothing
# 4. MLE with Dirichlet priors
# 
# Although this implementation is data-dependent and works for only one dataset,
# it should be relatively straighforward to apply it to other datasets.
#
# Version: 1.0, Date: 10 September 2024 with Laplace and Additive smoothing
# Version: 1.5, Date: 30 September 2025 with Dirichlet priors
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import numpy as np
import bnlearn as bn
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

VISUALISE_STRUCTURE = False
SMOOTHING_TECHNIQUE = "Dirichlet" # options={MLE,Laplace,Additive,Dirichlet}

# data loading using pandas
data = pd.read_csv('data/play_tennis-train.csv')
print("DATA:\n", data)

# definition of directed acyclic graph (predefined structure)
edges=[('PT', 'O'), ('PT', 'T'), ('PT', 'H'), ('PT', 'W'), ('O','T'), ('O','W'), ('T','H')]
DAG = bn.make_DAG(edges)
print("DAG:\n", DAG)

# visualise the structure above
if VISUALISE_STRUCTURE:
    G = nx.DiGraph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=15, font_weight='bold', arrows=True)
    plt.title('Directed Acyclic Graph (DAG)')
    plt.show()

# parameter learning using Maximum Likelihood Estimation with 
if SMOOTHING_TECHNIQUE == "Laplace": 
    model = bn.parameter_learning.fit(DAG, data, methodtype='bayes', smooth=1.0)
    pgm_model = model['model']

elif SMOOTHING_TECHNIQUE == "Additive": 
    l = 0.5  # adaptive smoothing factor
    adaptive_probs = {}
    pgm_model = DAG['model']

    # calculate adaptive smoothed probabilities
    for node in data.columns:
        counts = data[node].value_counts()
        N = counts.sum()
        J = data[node].nunique()
        adaptive_probs[node] = (counts + l) / (N + l * J)

    # build CPTs including smoothed probabilities
    for node in pgm_model.nodes():
        parents = list(pgm_model.get_parents(node))
        n_states = data[node].nunique()

        state_names = {node: list(data[node].unique())}
        for p in parents:
            state_names[p] = list(data[p].unique())

        if len(parents) == 0: # without parents
            values = adaptive_probs[node].values.reshape(n_states, 1)
            CPT = TabularCPD(variable=node, variable_card=n_states, values=values, state_names=state_names)
        else: # with parents
            parent_cards = [data[p].nunique() for p in parents]
            n_configs = np.prod(parent_cards)
            values = np.tile(adaptive_probs[node].values.reshape(n_states, 1), n_configs)
            CPT = TabularCPD(variable=node, variable_card=n_states,
                            values=values,
                            evidence=parents,
                            evidence_card=parent_cards,
                            state_names=state_names)
        pgm_model.add_cpds(CPT)

elif SMOOTHING_TECHNIQUE == "Dirichlet": 
    pseudo = {}; S = 5; epsilon = 0.5
    pgm_model = DAG['model']
    for node in pgm_model.nodes():
        counts = data[node].value_counts() # frequencies
        total = counts.sum() # sum of frequencies
        scaled = (counts / total * S) + epsilon
        scaled = np.array(scaled)
        parents = list(pgm_model.get_parents(node))
        if len(parents) == 0:
            n_configs = 1
        else:
            n_configs = np.prod([data[p].nunique() for p in parents])
        pseudo[node] = np.tile(scaled.reshape(-1,1), n_configs)

    pgm_model.fit(data, estimator=BayesianEstimator, prior_type='dirichlet', pseudo_counts=pseudo)

else:
    model = bn.parameter_learning.fit(DAG, data, methodtype="maximumlikelihood")
    pgm_model = model['model']
    print("WARNING: This choice of parameter learning (MLE) can involve zero probabilities!")

# probabilistic inference for the example covered in one of the last lecture slides of week 1
_evidence = {'W':'strong', 'O':'rain'}
infer = VariableElimination(pgm_model)
inference_result = infer.query(variables=['PT'], evidence=_evidence)
print(inference_result)