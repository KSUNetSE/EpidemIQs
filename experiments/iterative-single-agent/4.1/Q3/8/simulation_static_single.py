
import fastgemf as fg
import numpy as np
import scipy.sparse as sparse
import os

# Parameters based on previous code
beta_static = 0.01528
recovery_rate = 0.2
N = 1000
A_static = sparse.load_npz(os.path.join('output','network_static_agg.npz'))

SIR_schema = (
    fg.ModelSchema('SIR')
    .define_compartment(['S','I','R'])
    .add_network_layer('agg')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='agg', rate='beta')
)

SIR_config = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=beta_static, gamma=recovery_rate)
    .get_networks(agg=A_static)
)
ic_static = {'percentage':{'S':99,'I':1,'R':0}}

sim = fg.Simulation(SIR_config, initial_condition=ic_static, stop_condition={'time':100}, nsim=1)
sim.run()

time, state_count, *_ = sim.get_results()
import pandas as pd
data = pd.DataFrame({'time':time,'S':state_count[0],'I':state_count[1],'R':state_count[2]})
data.to_csv(os.path.join(os.getcwd(),'output','results-11.csv'),index=False)
