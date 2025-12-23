
import fastgemf as fg
import numpy as np, scipy.sparse as sparse, os, random
import networkx as nx

# load network
net_path=os.path.join(os.getcwd(),'output','network.npz')
G_csr=sparse.load_npz(net_path)
N=G_csr.shape[0]

# define SIR model schema
SIR_schema=(fg.ModelSchema('SIR').define_compartment(['S','I','R']).add_network_layer('contact').add_edge_interaction('infection',from_state='S',to_state='I',inducer='I',network_layer='contact',rate='beta').add_node_transition('recovery',from_state='I',to_state='R',rate='gamma'))

# configure model parameters
config=(fg.ModelConfiguration(SIR_schema).add_parameter(beta=1.0,gamma=1.0).get_networks(contact=G_csr))

print(config)
