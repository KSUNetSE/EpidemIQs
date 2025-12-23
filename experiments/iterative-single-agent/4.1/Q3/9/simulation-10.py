
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os

R0 = 3
recovery_rate = 0.1
agg_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'agg_network.npz'))
k_avg = 328.96
k2_avg = 108509.9
q = (k2_avg - k_avg) / k_avg
beta_static = R0 * recovery_rate / q

# SIR model schema
SIR_schema = (
    fg.ModelSchema("SIR")
        .define_compartment(['S', 'I', 'R'])
        .add_network_layer('agg_layer')
        .add_node_transition('recovery', from_state='I', to_state='R', rate='gamma')
        .add_edge_interaction('infection', from_state='S', to_state='I', inducer='I', network_layer='agg_layer', rate='beta')
)

SIR_instance = (
    fg.ModelConfiguration(SIR_schema)
        .add_parameter(beta=beta_static, gamma=recovery_rate)
        .get_networks(agg_layer=agg_csr)
)
initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}}

sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 60}, nsim=10)
sim.run()
time, state_count, *_ = sim.get_results()
# Save as CSV manually (workaround plotting)
import pandas as pd
data = {'time': time, 'S': state_count[0], 'I': state_count[1], 'R': state_count[2]}
df = pd.DataFrame(data)
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-10.csv'), index=False)
'finished'