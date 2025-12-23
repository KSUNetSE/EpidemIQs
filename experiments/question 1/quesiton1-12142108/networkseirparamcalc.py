
import numpy as np

def compute_network_seir_params(beta, sigma, gamma, k_mean, k2_mean):
    # Transmissibility
    T = beta / (beta + gamma)
    # Average excess degree
    kappa = (k2_mean - k_mean) / k_mean
    # Network reproduction number
    R0_network = T * kappa
    return T, kappa, R0_network

# Rates (given/clinical):
beta = 0.5  # per edge per day
sigma = 1.0/5.0  # E->I rate, per day
gamma = 1.0/7.0  # I->R rate, per day

# Network 1: ER
k_mean_ER = 9.95
k2_mean_ER = 108.7
T_ER, kappa_ER, R0_net_ER = compute_network_seir_params(beta, sigma, gamma, k_mean_ER, k2_mean_ER)

# Network 2: Scale-Free
k_mean_SF = 9.68
k2_mean_SF = 794.0
T_SF, kappa_SF, R0_net_SF = compute_network_seir_params(beta, sigma, gamma, k_mean_SF, k2_mean_SF)

# Initial condition (for population N=10,000)
N = 10000
E0 = 10
I0 = 1  # At least 1 initial infected per guidance; rest can be susceptible
S0 = N - E0 - I0
R0_ = 0
# Convert to percentages and round
S_pct = int(round(S0 / N * 100))
E_pct = int(round(E0 / N * 100))
I_pct = int(round(I0 / N * 100))
R_pct = 100 - (S_pct + E_pct + I_pct)

params_output = {
    "SEIR (ER network)": {"beta": beta, "sigma": sigma, "gamma": gamma, "type": "CTMC rates"},
    "SEIR (scale-free network)": {"beta": beta, "sigma": sigma, "gamma": gamma, "type": "CTMC rates"}}
initial_condition_desc = ["Initial exposed (E) nodes are randomly selected (0.1%), initial infectious (I) node is randomly selected (0.01%), remainder susceptible (S), no one initially recovered."]
initial_conditions = [{"S": S_pct, "E": E_pct, "I": I_pct, "R": R_pct}]

# Collect for return
return_vars = ["params_output", "initial_condition_desc", "initial_conditions", "T_ER", "kappa_ER", "R0_net_ER", "T_SF", "kappa_SF", "R0_net_SF"]
