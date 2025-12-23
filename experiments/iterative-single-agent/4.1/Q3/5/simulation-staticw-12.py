
import fastgemf as fg
import scipy.sparse as sparse
import os
import numpy as np
import pandas as pd

# SIR on the time-aggregated weighted network: instead of binary presence, use frequency-based edge weights (proxy for continuous contacts)
G_agg_csr = sparse.load_npz(os.path.join('output', 'network-agg.npz'))

# Use same beta, gamma
cfgw = (
    fg.ModelConfiguration(
        fg.ModelSchema('SIR')
        .define_compartment(['S', 'I', 'R'])
        .add_network_layer('contact')
        .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
        .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact', rate='beta')
    )
    .add_parameter(beta=1.6438356164383563, gamma=0.2)
    .get_networks(contact=G_agg_csr)
)

init_cond = {'percentage': {'S': 99, 'I': 1, 'R': 0}}

simw = fg.Simulation(cfgw, initial_condition=init_cond, stop_condition={'time': 100}, nsim=10)
simw.run()
timew, state_countw, *_ = simw.get_results()
resw = pd.DataFrame({'time': timew, 'S': state_countw[0], 'I': state_countw[1], 'R': state_countw[2]})
resw.to_csv(os.path.join('output','results-12.csv'), index=False)