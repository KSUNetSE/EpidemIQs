
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os
# Load the network
G_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))

# SIR ModelSchema
SIR_model_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('static_layer')
    .add_node_transition(
        name='recovery',
        from_state='I',
        to_state='R',
        rate='gamma'
    )
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='static_layer', rate='beta'
    )
)
# Parameters: choose values so R0 ~ 2.5, gamma = 0.2/day (rec. period 5 days), mean_k2 = 100.504, mean_k=10
mean_k = 10
mean_k2 = 100.504
gamma = 0.2
R0 = 2.5
q = (mean_k2 - mean_k)/mean_k
beta = R0 * gamma / q
# Parameter dict
params = {'beta': float(beta), 'gamma': gamma}
# Model instantiation
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=params['beta'], gamma=params['gamma'])
    .get_networks(static_layer=G_csr)
)
print('Used beta:', beta)
print('Mean excess degree q:', q)
print('SIR_model_schema', SIR_model_schema)
print('SIR_instance', SIR_instance)
return_vars=['beta', 'gamma', 'q']