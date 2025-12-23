
import fastgemf as fg
import os, numpy as np, scipy.sparse as sparse
# Load network
G_path = '/Users/hosseinsamaei/phd/gemf_llm/output/network.npz'
contact_csr = sparse.load_npz(G_path)

# Define SIR schema
SIR_schema = (
    fg.ModelSchema('SIR')
    .define_compartment(['S','I','R'])
    .add_network_layer('contact')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact', rate='beta')
)

results = {}

for j, (beta, R0_label) in enumerate([(0.009, 'moderate'), (0.0306, 'high')], start=1):
    SIR_instance = (
        fg.ModelConfiguration(SIR_schema)
        .add_parameter(beta=beta, gamma=0.1)
        .get_networks(contact=contact_csr)
    )
    # Initial condition: 10 infected hubs (highest degree) rest susceptible
    degrees = np.array(contact_csr.sum(axis=1)).flatten()
    hubs = np.argsort(-degrees)[:10]
    X0 = np.zeros(contact_csr.shape[0], dtype=int)  # S=0
    X0[hubs] = 1  # I=1
    initial_cond = {'exact': X0}

    sim = fg.Simulation(SIR_instance, initial_condition=initial_cond, stop_condition={'time':160}, nsim=1)
    sim.run()
    # Save plot
    plot_path = os.path.join(os.getcwd(), 'output', f'results-1{j}.png')
    sim.plot_results(show_figure=False, save_figure=True, save_path=plot_path)

    # Retrieve data
    time, state_count, *_ = sim.get_results()
    data = {
        't': time,
        'S': state_count[0],
        'I': state_count[1],
        'R': state_count[2],
    }
    import pandas as pd
    df = pd.DataFrame(data)
    csv_path = os.path.join(os.getcwd(), 'output', f'results-1{j}.csv')
    df.to_csv(csv_path, index=False)
    results[j] = {'R0_label': R0_label, 'beta': beta, 'csv': csv_path, 'png': plot_path}

results