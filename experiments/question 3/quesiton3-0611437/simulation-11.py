
import fastgemf as fg
import scipy.sparse as sparse
import pandas as pd
import numpy as np
import os

# (i,j) = (1,1): Static Weighted Network
# 1. Load the network
G_csr = sparse.load_npz('/Users/hosseinsamaei/phd/gemf_llm/output/static-weighted-network.npz')

# 2. Define SIR schema
sir_schema = (
    fg.ModelSchema('SIR')
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(
        name='recovery',
        from_state='I',
        to_state='R',
        rate='mu',
    )
    .add_edge_interaction(
        name='infection',
        from_state='S',
        to_state='I',
        inducer='I',
        network_layer='contact_network_layer',
        rate='beta',
    )
)

# 3. Specify parameters (from task)
params = {'beta': 82.08281, 'mu': 1.0}

# 4. Model config
sir_instance = (
    fg.ModelConfiguration(sir_schema)
    .add_parameter(**params)
    .get_networks(contact_network_layer=G_csr)
)

# 5. Initial condition: 1% infected, 99% susceptible, 0% recovered (random)
# FastGEMF uses percentage specification
initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}}

# 6. Simulation: sr=100 realizations, run to t=365
sim = fg.Simulation(sir_instance, initial_condition=initial_condition, stop_condition={'time': 365}, nsim=100)
sim.run()

# 7. Save plot
os.makedirs(os.path.join(os.getcwd(), 'output'), exist_ok=True)
sim.plot_results(show_figure=False, save_figure=True,
                 save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))

# 8. Save results
results = sim.get_results()
time, state_count, *_ = results
simulation_results = {}
simulation_results['time'] = time
for i in range(state_count.shape[0]):
    simulation_results[f'{sir_schema.compartments[i]}'] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)
