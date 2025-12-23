
def calculate_network_sir_parameters(R0_wm, T, mean_degree):
    gamma = 1.0 / T
    beta_edge = (R0_wm * gamma) / mean_degree
    # Tree-like approximation for local spreading
    T_transmit = beta_edge / (beta_edge + gamma)
    R0_network = mean_degree * T_transmit
    return {
        'beta_edge': round(beta_edge, 5),
        'gamma': round(gamma, 5),
        'implied_R0_network': round(R0_network, 4)
    }

# Define input values
R0_wm = 2.0
T = 4.0
ws_mean_degree = 6.00
er_mean_degree = 5.95

params_ws = calculate_network_sir_parameters(R0_wm, T, ws_mean_degree)
params_er = calculate_network_sir_parameters(R0_wm, T, er_mean_degree)

# Initial condition (percentages): single infectious out of 1000 nodes
# S = 999/1000 = 99.9%, I = 0.1%, R = 0%. Need to enforce at least 1% I for model relevance.
IC = {'S': 99, 'I': 1, 'R': 0}

params_output = {
    'WattsStrogatz': {'beta_edge': params_ws['beta_edge'], 'gamma': params_ws['gamma']},
    'ErdosRenyi': {'beta_edge': params_er['beta_edge'], 'gamma': params_er['gamma']}
}
implied_R0s = {
    'WattsStrogatz': params_ws['implied_R0_network'],
    'ErdosRenyi': params_er['implied_R0_network']
}

IC_desc = ["Single randomly chosen infectious node, rest susceptible (no initial removed)"]

return_vars = ['params_output', 'implied_R0s', 'IC', 'IC_desc']
