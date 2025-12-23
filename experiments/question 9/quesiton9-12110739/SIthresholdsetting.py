
import numpy as np
# 1. Model Compartments: ["S", "I"], SI-threshold, transition: S->I if m/k >= φ
# 2. All transitions are deterministic and discrete (no rates needed). Only parameter: φ = B/E

# 3. Network parameters (from context):
N = 500
E = 100.0
mean_deg_ER = 19.56
mean_deg_BA = 5.96

# 4. Cascade threshold predictions:
phi_c_ER = 1/mean_deg_ER # ~0.051 for ER
phi_c_BA = 1/mean_deg_BA # ~0.168 for BA (approximate, BA's real threshold is lower due to hubs)

# Range to test for φ:
phi_values_ER = np.round(np.linspace(0.01, 0.10, 10), 3)
phi_values_BA = np.round(np.linspace(0.01, 0.20, 20), 3)

# For output, prepare a parameter list for each network
params_ER = [{"φ": float(phi), "E": E, "B": float(phi)*E} for phi in phi_values_ER]
params_BA = [{"φ": float(phi), "E": E, "B": float(phi)*E} for phi in phi_values_BA]

# 5. Initial conditions (ensure at least 1% seed, i.e., 5/500=1%)
initial_fraction = 1 # as percentage
I_num = max(1, int(N*initial_fraction/100)) # at least 1 node
S_num = N - I_num

# Both random and targeted (hub) shocks
initconds_random = {"S": S_num, "I": I_num}
initconds_targeted = {"S": S_num, "I": I_num} # just descriptive, actual selection is in simulation, not here

# List for different scenario setups
parameter_scenarios = {
    "ER": params_ER,
    "BA": params_BA
}
initial_conditions = [initconds_random, initconds_targeted]
initial_condition_desc = [
    "Random: 1% of nodes randomly defaulted, rest susceptible",
    "Targeted: 1% of top-degree (hub) nodes defaulted, rest susceptible",
]

# Aggregate final output
output_vars = [
    "parameter_scenarios", "initial_conditions", "initial_condition_desc", "phi_c_ER", "phi_c_BA"
]
return_vars = output_vars
