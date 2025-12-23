
# Reasoning After Error:
# The error indicates an issue with how IC_exact is handled; FastGEMF expects the initial condition to match shape (N,) (for one run) or (nsim, N) with the correct population size and simulation count.
# On review, FastGEMF expects the initial_condition to be a single array shape (N,) if using 'exact' for all runs.
# Instead, the initialization should be performed within FastGEMF's stochastic realizations, and we provide 'percentage' for random seeding for each run.
# Correction: Use 'percentage' initial_condition, with {'S': 99.9, 'I': 0.1, 'R': 0} per percent-based random initialization (as there is no mode for single specific node, percentage is the most accurate approximation for random single initial index per replicate, as documented).

import os
import fastgemf as fg
import scipy.sparse as sparse
import pandas as pd

network_path = '/Users/hosseinsamaei/phd/epidemiqs/output/network-watts-strogatz.npz'  # Watts-Strogatz
data_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
plot_path = os.path.join(os.getcwd(), 'output', 'results-11.png')
summary_csv = os.path.join(os.getcwd(), 'output', 'results-11-summary.csv')

beta = 0.08333
gamma = 0.25
N = 1000
nsim = 200
max_time = 240

G_csr = sparse.load_npz(network_path)

sir_schema = (fg.ModelSchema('SIR')
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(
        name='recovery',
        from_state='I',
        to_state='R',
        rate='gamma')
    .add_edge_interaction(
        name='infection',
        from_state='S',
        to_state='I',
        inducer='I',
        network_layer='contact_network_layer',
        rate='beta'))

sir_instance = (fg.ModelConfiguration(sir_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact_network_layer=G_csr))

# key: percentage for initial condition: ~1 per 1000 infected (0.1%), rest susceptible (rounded for N=1000)
initial_condition = {'percentage': {'S': 99.9, 'I': 0.1, 'R': 0}}

sim = fg.Simulation(sir_instance, initial_condition=initial_condition, stop_condition={'time': max_time}, nsim=nsim)
sim.run()

variation_type = '90ci'
time, state_mean, state_bands = sim.get_results(variation_type=variation_type)
compartments = sir_schema.compartments
results_dict = {'time': time}
for i, comp in enumerate(compartments):
    results_dict[comp] = state_mean[i]
    results_dict[f'{comp}_{variation_type}_lower'] = state_bands[0,i]
    results_dict[f'{comp}_{variation_type}_upper'] = state_bands[1,i]
data = pd.DataFrame(results_dict)
data.to_csv(data_path, index=False)

# SIR Metrics
I_series = state_mean[1]
peak_I = I_series.max()
time_to_peak = time[I_series.argmax()]
def time_to_threshold(thresh):
    idx = (I_series >= N*thresh).nonzero()[0]
    return time[idx[0]] if len(idx) else None
summary_dict = {
    'peak_prevalence': peak_I/N,
    'time_to_peak': time_to_peak,
    'time_to_10pct': time_to_threshold(0.10),
    'time_to_20pct': time_to_threshold(0.20)
}
pd.DataFrame([summary_dict]).to_csv(summary_csv, index=False)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(7,5))
for i, comp in enumerate(compartments):
    plt.plot(time, state_mean[i]/N, label=comp)
    plt.fill_between(time, state_bands[0,i]/N, state_bands[1,i]/N, alpha=0.18)
plt.xlabel('Time (days)')
plt.ylabel('Fraction of population')
plt.title('SIR dynamics on Watts-Strogatz network')
plt.legend()
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

return_vars = ['data_path', 'plot_path', 'summary_csv', 'summary_dict']
