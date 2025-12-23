
# Plan: Setup and run FastGEMF simulation for SEIR on the two networks.
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os
import pandas as pd

output_dir = os.path.join(os.getcwd(), 'output')

# Define SEIR schema
SEIR_schema = (
    fg.ModelSchema("SEIR")
    .define_compartment(['S', 'E', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(name='exposed_to_infectious', from_state='E', to_state='I', rate='alpha')
    .add_node_transition(name='infect_to_recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='E', inducer='I',
        network_layer='contact_network_layer', rate='beta')
)

### Load networks
A_hom = sparse.load_npz(os.path.join(output_dir, 'network_homogeneous.npz'))
A_het = sparse.load_npz(os.path.join(output_dir, 'network_heterogeneous.npz'))

# Parameters
params_hom = {'beta': 0.3571428571428571, 'alpha': 0.3333333333333333, 'gamma': 0.14285714285714285}
params_het = {'beta': 0.017785820821887247, 'alpha': 0.3333333333333333, 'gamma': 0.14285714285714285}

# Initial condition: 10 infected, 990 susceptible
ic = {'percentage': {'S': 98, 'E': 0, 'I': 2, 'R': 0}}  # For n=1000

# Homogeneous network simulation
SEIRcfg_hom = (
    fg.ModelConfiguration(SEIR_schema)
    .add_parameter(**params_hom)
    .get_networks(contact_network_layer=A_hom)
)
sim_hom = fg.Simulation(SEIRcfg_hom, initial_condition=ic, stop_condition={'time':150}, nsim=10)
sim_hom.run()
sim_hom.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(output_dir,'results-1-1.png'))
# Save CSV
t_hom, states_hom, *_ = sim_hom.get_results()
df_hom = pd.DataFrame({'time': t_hom, 'S': states_hom[0], 'E': states_hom[1], 'I': states_hom[2], 'R': states_hom[3]})
df_hom.to_csv(os.path.join(output_dir,'results-1-1.csv'), index=False)

# Heterogeneous network simulation
SEIRcfg_het = (
    fg.ModelConfiguration(SEIR_schema)
    .add_parameter(**params_het)
    .get_networks(contact_network_layer=A_het)
)
sim_het = fg.Simulation(SEIRcfg_het, initial_condition=ic, stop_condition={'time':150}, nsim=10)
sim_het.run()
sim_het.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(output_dir,'results-1-2.png'))
t_het, states_het, *_ = sim_het.get_results()
df_het = pd.DataFrame({'time': t_het, 'S': states_het[0], 'E': states_het[1], 'I': states_het[2], 'R': states_het[3]})
df_het.to_csv(os.path.join(output_dir,'results-1-2.csv'), index=False)

# Return outputs
results_info = {
    'homogeneous': {'csv': os.path.join(output_dir,'results-1-1.csv'), 'png': os.path.join(output_dir,'results-1-1.png')},
    'heterogeneous': {'csv': os.path.join(output_dir,'results-1-2.csv'), 'png': os.path.join(output_dir,'results-1-2.png')}
}
results_info
