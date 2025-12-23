
# Set up parameters for the ER and BA networks for network-based CTMC SEIR simulations
# Population size
N = 5000

# Input rates (per day)
beta_wm = 0.3  # Given as per-contact in well-mixed model
sigma = 1/3    # Exposed to infectious, per day
sigma = round(sigma, 3)  # 0.333
gamma = 1/4    # Infectious to recovered, per day
gamma = round(gamma, 3)  # 0.25

# Network means
k_ER = 7.84
k_BA = 7.99
k2_ER = 69.4
k2_BA = 167.2

# -- Compute per-edge transmission hazards --
beta_edge_ER = beta_wm / k_ER
beta_edge_BA = beta_wm / k_BA

# -- Compute effective/expected R0 for each network --
R0_ER = beta_wm / gamma  # ODE style
R0_network_BA = (beta_wm / gamma) * ((k2_BA - k_BA)/k_BA)
R0_network_ER = (beta_wm / gamma) * ((k2_ER - k_ER)/k_ER)
# Note: Network-level R0 for ER is slightly lower than ODE value due to finite degree variance

# -- Initial conditions: 1% exposed, rest susceptible --
frac_S = round((N - N//100)/N * 100)
frac_E = round((N//100)/N * 100)
frac_I = 0
frac_R = 0

# Ensure sum is 100, fix by adjusting S accordingly if needed
sum_ = frac_S + frac_E + frac_I + frac_R
if sum_ > 100:
    frac_S -= (sum_ - 100)
elif sum_ < 100:
    frac_S += (100 - sum_)

initial_conditions = [{'S': frac_S, 'E': frac_E, 'I': frac_I, 'R': frac_R}]

return_vars = [
    'beta_edge_ER', 'beta_edge_BA', 'sigma', 'gamma', 'R0_ER',
    'R0_network_ER', 'R0_network_BA', 'initial_conditions'
]
