
# i=15
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import os
beta1, delta1, beta2, delta2 = [0.10449, 1.0, 0.11855, 1.0]
layerA_path = os.path.join(os.getcwd(), 'output', 'network-layerA.npz')
layerB_path = os.path.join(os.getcwd(), 'output', 'network-layerB.npz')
G_A = sparse.load_npz(layerA_path)
G_B = sparse.load_npz(layerB_path)
N = G_A.shape[0]
SIS2_model = (
    fg.ModelSchema('CompetitiveSIS')
    .define_compartment(['S', 'I1', 'I2'])
    .add_network_layer('layerA')
    .add_edge_interaction(
        name='infection1', from_state='S', to_state='I1', inducer='I1', network_layer='layerA', rate='beta1')
    .add_network_layer('layerB')
    .add_edge_interaction(
        name='infection2', from_state='S', to_state='I2', inducer='I2', network_layer='layerB', rate='beta2')
    .add_node_transition(
        name='rec1', from_state='I1', to_state='S', rate='delta1')
    .add_node_transition(
        name='rec2', from_state='I2', to_state='S', rate='delta2')
)
sr = 50
network_details = f"LayerA: N={N}, edges={G_A.nnz//2}; LayerB: N={N}, edges={G_B.nnz//2}"
out_sim_details = []
out_paths = {}
out_plots = {}
for j in range(3):
    X0 = np.zeros(N, dtype=int)
    rng = np.random.default_rng(202406 + 100*15 + 10*j)
    if j == 0:
        I1_inds = rng.choice(N, size=10, replace=False)
        X0[I1_inds] = 1
    elif j == 1:
        I2_inds = rng.choice(N, size=10, replace=False)
        X0[I2_inds] = 2
    elif j == 2:
        I_seeds = rng.choice(N, size=20, replace=False)
        I1_inds = I_seeds[:10]
        I2_inds = I_seeds[10:]
        X0[I1_inds] = 1
        X0[I2_inds] = 2
    initial_condition = {'exact': X0}
    model_instance = (
        fg.ModelConfiguration(SIS2_model)
        .add_parameter(beta1=beta1, delta1=delta1, beta2=beta2, delta2=delta2)
        .get_networks(layerA=G_A, layerB=G_B)
    )
    sim = fg.Simulation(model_instance, initial_condition=initial_condition, stop_condition={'time': 200}, nsim=sr)
    sim.run()
    plot_path = os.path.join(os.getcwd(), 'output', f'results-15{j}.png')
    sim.plot_results(show_figure=False, save_figure=True, save_path=plot_path)
    time, state_count, *_ = sim.get_results()
    sim_results = { 'time': time }
    for idx, comp in enumerate(SIS2_model.compartments):
        sim_results[comp] = state_count[idx, :]
    csv_path = os.path.join(os.getcwd(), 'output', f'results-15{j}.csv')
    pd.DataFrame(sim_results).to_csv(csv_path, index=False)
    if j == 0:
        ic_desc = '1% I1, rest S'
    elif j == 1:
        ic_desc = '1% I2, rest S'
    else:
        ic_desc = '1% I1, 1% I2, rest S'
    scenario_string = f'Competitive SIS multiplex, param set 15 (b1={beta1}, d1={delta1}, b2={beta2}, d2={delta2}), IC-{j} ({ic_desc}), 50 stoch. realizations, {network_details}, T=200.'
    out_sim_details.append(scenario_string)
    out_paths[csv_path] = f'Raw time series for {scenario_string}'
    out_plots[plot_path] = network_details
    del sim, X0
def get_response():
    return dict(
        simulation_details=out_sim_details,
        stored_result_path=out_paths,
        Plot_path=out_plots,
        success_of_simulation=True,
        reasoning_info="i=15, all ICs complete. ALL SCENARIOS DONE."
    )
api_results = get_response()