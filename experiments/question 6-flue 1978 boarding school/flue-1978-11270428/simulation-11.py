
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the SBM network
network_path = os.path.join(os.getcwd(), 'output', 'boarding-school-sbm-network.npz')
if not os.path.isfile(network_path):
    raise FileNotFoundError(f'Expected SBM network file at {network_path}!')
G_csr = sparse.load_npz(network_path)

# 2. Load empirical data (now from CSV, due to limitations)
data_path = os.path.join(os.getcwd(), 'output', 'cases_data.csv')
case_df = pd.read_csv(data_path)  # expects columns: 'Day', 'Confined_to_bed', 'Convalescents'
empirical_days = case_df['Day'].values
empirical_Q = case_df['Confined_to_bed'].values
empirical_R = case_df['Convalescents'].values

# 3. Define SEIQR model via FastGEMF schema
SEIQR_schema = (
    fg.ModelSchema('SEIQR')
    .define_compartment(['S', 'E', 'I', 'Q', 'R'])
    .add_network_layer('school_network')
    .add_edge_interaction('infection', from_state='S', to_state='E', inducer='I', network_layer='school_network', rate='beta')
    .add_node_transition('incubation', from_state='E', to_state='I', rate='sigma')
    .add_node_transition('hidden_to_quarantined', from_state='I', to_state='Q', rate='tau')
    .add_node_transition('quarantine_to_recovered', from_state='Q', to_state='R', rate='gamma')
)

# 4. Parameters (baseline)
params = dict(beta=0.039, sigma=1.0, tau=1.11, gamma=0.5)
SEIQR_instance = (
    fg.ModelConfiguration(SEIQR_schema)
    .add_parameter(**params)
    .get_networks(school_network=G_csr)  # assign the SBM
)
_ = SEIQR_instance

# 5. Initial condition -- exact vector: S=762, I=1, rest zero
N = G_csr.shape[0]
X0 = np.zeros(N, dtype=int)  # 0=S
init_I = np.random.choice(N, size=1, replace=False)
X0[init_I] = 2  # I = index 2
initial_condition = {'exact': X0}

# 6. Run simulation (baseline)
nsim = 400  # adequate repetitions for statistics
stop_time = 45
variation_type = '90ci'

sim = fg.Simulation(SEIQR_instance, initial_condition=initial_condition,
                    stop_condition={'time': stop_time}, nsim=nsim)
sim.run()

# 7. Extract and save results
(time, statecounts_mean, (lower, upper)) = sim.get_results(variation_type=variation_type)

# Save results (csv)
results_df = pd.DataFrame({'time': time})
comp_names = SEIQR_schema.compartments
for k, cname in enumerate(comp_names):
    results_df[cname] = statecounts_mean[k]
    results_df[f'{cname}_90ci_lower'] = lower[k]
    results_df[f'{cname}_90ci_upper'] = upper[k]
results_csv_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
results_df.to_csv(results_csv_path, index=False)

# 8. Plot simulation vs empirical (Q and R)
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
plt.title('SEIQR school SBM: simulated vs. observed outbreak\n"Q = confined to bed", "R = convalescents"', fontsize=14)
plt.legend()
plt.tight_layout()
plot_path = os.path.join(os.getcwd(), 'output', 'results-11.png')
plt.savefig(plot_path)
plt.close()

# 9. Compute summary statistics
final_attack_rate = statecounts_mean[4, -1] / N
peak_Q_idx = np.argmax(statecounts_mean[3])
peak_Q_time = time[peak_Q_idx]
peak_Q_value = statecounts_mean[3, peak_Q_idx]
R0_model = params['beta'] / params['tau'] * ((2233.5-46.8)/46.8)  # As per param computation, for reporting
# Prepare summary for reporting
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
summary_path = os.path.join(os.getcwd(), 'output', 'results-11-summary.csv')
pd.DataFrame([summary]).to_csv(summary_path, index=False)
