
# Step 2: Variation - beta sensitivity, test lower and higher betas:
# Repeat simulation for beta=0.35 and beta=0.42 (per protocol). Save as results-12, results-13
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import os

N = 763
network_path = os.path.join(os.getcwd(), 'output', 'england-boarding-sbm.npz')
G_csr = sparse.load_npz(network_path)

SEIBR_schema = (
    fg.ModelSchema('SEIBR')
    .define_compartment(['S', 'E', 'I', 'B', 'R'])
    .add_network_layer('contact_network')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='E', inducer='I',
        network_layer='contact_network', rate='beta'
    )
    .add_node_transition(name='latent_exit', from_state='E', to_state='I', rate='sigma')
    .add_node_transition(name='symptom_isolation', from_state='I', to_state='B', rate='gamma1')
    .add_node_transition(name='recover', from_state='B', to_state='R', rate='gamma2')
)

# Sensitivity grid:
betas = [0.35, 0.42]
results_summary = {}
for idx, beta in enumerate(betas):
    params = {"beta": beta, "sigma": 2.0, "gamma1": 0.7, "gamma2": 0.5}
    SEIBR_model = (
        fg.ModelConfiguration(SEIBR_schema)
        .add_parameter(**params)
        .get_networks(contact_network=G_csr)
    )
    # Initial condition: one random node E=1 as above for comparability
    X0 = np.zeros(N, dtype=int)
    seed = np.random.default_rng(seed=42).integers(0, N)
    X0[seed] = 1
    initial_condition = {"exact": X0}
    sr = 100
    sim = fg.Simulation(SEIBR_model, initial_condition=initial_condition, stop_condition={"time": 50}, nsim=sr)
    sim.run()
    variation_type = "90ci"
    time, state_mean, [state_low, state_high] = sim.get_results(variation_type=variation_type)
    out_dict = {'time': time}
    comps = ['S', 'E', 'I', 'B', 'R']
    for i, c in enumerate(comps):
        out_dict[c] = state_mean[i]
        out_dict[c+"_lower"] = state_low[i]
        out_dict[c+"_upper"] = state_high[i]
    C = np.diff(state_mean[4], prepend=0)
    out_dict['C'] = C
    data = pd.DataFrame(out_dict)
    result_path = os.path.join(os.getcwd(), 'output', f'results-1{2+idx}.csv')
    data.to_csv(result_path, index=False)
    sim.plot_results(time, state_mean, variation_type=variation_type, show_figure=False, save_figure=True,
        title=f"SEIBR SBM simulation: Beta={beta}",
        save_path=os.path.join(os.getcwd(), 'output', f'results-1{2+idx}.png'))

    attack_rate = state_mean[4,-1] / N
    peak_B_idx = np.argmax(state_mean[3])
    peak_B_time = time[peak_B_idx]
    peak_B_value = state_mean[3, peak_B_idx]
    results_summary[f"beta={beta}"] = {
        'result_path': result_path,
        'attack_rate': attack_rate,
        'peak_B_time': peak_B_time,
        'peak_B_value': peak_B_value,
    }

return_vars = ['results_summary']
