
# Sensitivity: test with Erdos-Renyi (same mean degree, see difference)
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

N = 763
mean_degree = 46.8
p_edge = mean_degree/(N-1)
G_er = nx.erdos_renyi_graph(N, p_edge, seed=44)
G_csr = sparse.csr_matrix(nx.to_numpy_array(G_er))

# Save for reproducibility if needed
results_dir = os.path.join(os.getcwd(), 'output')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Empirical data as before
case_df = pd.read_csv(os.path.join(results_dir, 'cases_data.csv'))
empirical_days = case_df['Day'].values
empirical_Q = case_df['Confined_to_bed'].values
empirical_R = case_df['Convalescents'].values

# SEIQR schema
SEIQR_schema = (
    fg.ModelSchema('SEIQR')
    .define_compartment(['S', 'E', 'I', 'Q', 'R'])
    .add_network_layer('er_network')
    .add_edge_interaction('infection', from_state='S', to_state='E', inducer='I', network_layer='er_network', rate='beta')
    .add_node_transition('incubation', from_state='E', to_state='I', rate='sigma')
    .add_node_transition('hidden_to_quarantined', from_state='I', to_state='Q', rate='tau')
    .add_node_transition('quarantine_to_recovered', from_state='Q', to_state='R', rate='gamma')
)

params = dict(beta=0.039, sigma=1.0, tau=1.11, gamma=0.5)
SEIQR_instance = (
    fg.ModelConfiguration(SEIQR_schema)
    .add_parameter(**params)
    .get_networks(er_network=G_csr)
)
X0 = np.zeros(N, dtype=int)
init_I = np.random.choice(N, size=1, replace=False)
X0[init_I] = 2
initial_condition = {'exact': X0}

nsim = 400
stop_time = 45
variation_type = '90ci'
sim = fg.Simulation(SEIQR_instance, initial_condition=initial_condition,
                    stop_condition={'time': stop_time}, nsim=nsim)
sim.run()
(time, statecounts_mean, (lower, upper)) = sim.get_results(variation_type=variation_type)

results_df = pd.DataFrame({'time': time})
comp_names = SEIQR_schema.compartments
for k, cname in enumerate(comp_names):
    results_df[cname] = statecounts_mean[k]
    results_df[f'{cname}_90ci_lower'] = lower[k]
    results_df[f'{cname}_90ci_upper'] = upper[k]
results_csv_path = os.path.join(os.getcwd(), 'output', 'results-14.csv')
results_df.to_csv(results_csv_path, index=False)

plt.figure(figsize=(10, 7))
plt.fill_between(time, lower[3], upper[3], color='orange', alpha=0.25, label='Q (confined to bed) 90% CI')
plt.plot(time, statecounts_mean[3], label='Q simulated mean', color='orange')
plt.scatter(empirical_days, empirical_Q, c='brown', label='Q observed', marker='s')
plt.fill_between(time, lower[4], upper[4], color='mediumseagreen', alpha=0.22, label='R (convalescent) 90% CI')
plt.plot(time, statecounts_mean[4], label='R simulated mean', color='green')
plt.scatter(empirical_days, empirical_R, c='darkgreen', label='R observed', marker='o')
plt.xlabel('Time (day)', fontsize=13)
plt.ylabel('Number of students', fontsize=13)
plt.grid()
plt.title('SEIQR Erdos-Renyi: simulated vs. observed')
plt.legend()
plt.tight_layout()
plot_path = os.path.join(os.getcwd(), 'output', 'results-14.png')
plt.savefig(plot_path)
plt.close()

final_attack_rate = statecounts_mean[4, -1]/N
peak_Q_idx = np.argmax(statecounts_mean[3])
peak_Q_time = time[peak_Q_idx]
peak_Q_value = statecounts_mean[3, peak_Q_idx]
R0_model = params['beta'] / params['tau'] * ((mean_degree**2-mean_degree)/mean_degree)
summary = {
    'final_attack_rate_model': final_attack_rate,
    'empirical_final_attack_rate': empirical_R[-1] / N,
    'peak_Q_time': float(peak_Q_time),
    'peak_Q_size': float(peak_Q_value),
    'empirical_peak_Q_time': float(empirical_days[np.argmax(empirical_Q)]),
    'empirical_peak_Q_size': float(np.max(empirical_Q)),
    'model_R0': R0_model,
    'empirical_R0': 1.66,
    'N': N
}
summary_path = os.path.join(os.getcwd(), 'output', 'results-14-summary.csv')
pd.DataFrame([summary]).to_csv(summary_path, index=False)
