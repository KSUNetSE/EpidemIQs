
import os, numpy as np, networkx as nx, scipy.sparse as sparse, fastgemf as fg, random, pandas as pd
# Load network
csr=sparse.load_npz(os.path.join(os.getcwd(),'output','network.npz'))
N=csr.shape[0]

# Build SIR model schema
SIR_model_schema=(
    fg.ModelSchema('SIR')
    .define_compartment(['S','I','R'])
    .add_network_layer('contact')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact', rate='beta')
)
# parameters
beta=1.01
gamma=1.0
config=(
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact=csr)
)
print('Model configured')
