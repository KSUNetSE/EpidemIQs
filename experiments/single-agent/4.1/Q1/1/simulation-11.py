
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os
import pandas as pd

G_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))
# Model schema (SIR)
SIR_model_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('static_layer')
    .add_node_transition(
        name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='static_layer', rate='beta')
)
# Parameters from previous calculation
params = {'beta': 0.055246176964554056, 'gamma': 0.2}
# Model configuration
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=params['beta'], gamma=params['gamma'])
    .get_networks(static_layer=G_csr)
)
# Initial condition: 99% S, 1% I, 0% R (randomly spread)
N = G_csr.shape[0]
num_I = int(0.01*N)
num_S = N - num_I
init_cond = {'percentage': {'S': int(100*num_S/N), 'I': int(100*num_I/N), 'R': 0}}
# Simulation
sim = fg.Simulation(SIR_instance, initial_condition=init_cond, stop_condition={'time': 100}, nsim=10)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))
# Save CSV results
time, counts, *_ = sim.get_results()
res_df = pd.DataFrame({'time': time, 'S': counts[0], 'I': counts[1], 'R': counts[2]})
res_df.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)
return_vars=['res_df.columns']
import fastgemf as fg
import numpy as np
from scipy import sparse
import os
import pandas as pd

# Load the network
network_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))

# 1. Define the SIR mechanistic model over network as per our setup
SIR_model_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(
        name='recovery',
        from_state='I',
        to_state='R',
        rate='gamma'
    )
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='contact_network_layer', rate='beta'
    )
)

# 2. Set up model instance with parameters
parameters = {'beta': 0.05045, 'gamma': 0.2}
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(**parameters)
    .get_networks(contact_network_layer=network_csr)
)

# 3. Initial condition: percentage-based (random assignment: S = 99%, I = 1%, R = 0)
initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}}

# 4. Simulate for 180 days, 10 stochastic runs
sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 180}, nsim=10)
sim.run()

# 5. Plot and save the results
png_path = os.path.join(os.getcwd(), 'output', 'results-11.png')
sim.plot_results(show_figure=False, save_figure=True, save_path=png_path)

# 6. Save results as CSV
results = {}
time, state_count, *_ = sim.get_results()
results['time'] = time
for i in range(state_count.shape[0]):
    results[SIR_model_schema.compartments[i]] = state_count[i, :]
data = pd.DataFrame(results)
csv_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
data.to_csv(csv_path, index=False)
csv_path, png_path