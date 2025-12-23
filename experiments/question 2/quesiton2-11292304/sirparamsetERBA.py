
# Parameter extraction and mapping for ER and BA networks
# Given parameters (canonical):
R0 = 3.0
beta_wm = 0.3  # per day, well-mixed transmission rate
gamma = 0.1    # per day, recovery rate
N = 1000  # population

# ER Network
er_k_mean = 10.064
# BA Network
ba_k_mean = 9.99

# Map to per-edge hazard for network (assuming unweighted, undirected)
def per_edge_beta(R0, gamma, k_mean):
    return (R0 * gamma) / k_mean

beta_edge_er = per_edge_beta(R0, gamma, er_k_mean)
beta_edge_ba = per_edge_beta(R0, gamma, ba_k_mean)

# Compute T = beta_edge / (beta_edge + gamma), network reproduction numbers
T_er = beta_edge_er / (beta_edge_er + gamma)
T_ba = beta_edge_ba / (beta_edge_ba + gamma)

R0_er_network = er_k_mean * T_er
R0_ba_network = ba_k_mean * T_ba

# Initial condition as integer percentages (I0=1, S0=999, N=1000)
I0 = 1
S0 = 999
R0_0 = 0

S_percent = round(S0 * 100 / N)
I_percent = round(I0 * 100 / N)
R_percent = round(R0_0 * 100 / N)
# Correction (due to rounding, ensure sum==100)
remainder = 100 - (S_percent + I_percent + R_percent)
S_percent += remainder # ensures sum=100

return_vars = [
    "beta_edge_er","beta_edge_ba","gamma",
    "T_er","T_ba","R0_er_network","R0_ba_network",
    "S_percent","I_percent","R_percent"
]
