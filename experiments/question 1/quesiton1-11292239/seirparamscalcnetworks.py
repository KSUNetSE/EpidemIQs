
def get_seir_network_parameters(beta, sigma, gamma, mean_degree, second_moment_degree):
    """Calculate network-specific SEIR parameters and diagnostics."""
    # 1. Per-edge transmission rate for network-based CTMC
    beta_edge = beta / mean_degree
    # 2. Diagnostic: implied network R0
    #   - For ER: R0_network = beta_edge × mean_degree / gamma
    #   - For BA:  R0_network = beta_edge × (second_moment_degree - mean_degree) / (gamma * mean_degree)
    R0_er = (beta_edge * mean_degree) / gamma
    if mean_degree != 0:
        R0_ba = (beta_edge * (second_moment_degree - mean_degree)) / (gamma * mean_degree)
    else:
        R0_ba = None
    # 3. Return all in a dict
    return {
        "beta_edge": beta_edge, "sigma": sigma, "gamma": gamma,
        "R0_er": R0_er, "R0_ba": R0_ba
    }

# Problem constants
beta = 1.1 # per day
gamma = 0.43 # per day
sigma = 0.2 # per day

# Network stats
mean_degree_er = 8.07
second_moment_er = 73.27
mean_degree_ba = 7.98
second_moment_ba = 146.48

# Compute parameter sets
params_er = get_seir_network_parameters(beta, sigma, gamma, mean_degree_er, second_moment_er)
params_ba = get_seir_network_parameters(beta, sigma, gamma, mean_degree_ba, second_moment_ba)

params_er_simple = {"beta_edge": round(params_er["beta_edge"], 4), "sigma": sigma, "gamma": gamma}
params_ba_simple = {"beta_edge": round(params_ba["beta_edge"], 4), "sigma": sigma, "gamma": gamma}

# Diagnostics for reasoning
r0network_er = round(params_er["R0_er"], 2)
r0network_ba = round(params_ba["R0_ba"], 2)

return_vars = ["params_er_simple","params_ba_simple", "r0network_er", "r0network_ba"]