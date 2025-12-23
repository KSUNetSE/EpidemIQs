
# Step 1: Compute initial condition for the two vaccination scenarios
N = 10000
initial_infecteds = 1

# (1) Random vaccination
target_random_vacc_frac = 0.75
n_r_rand = round(target_random_vacc_frac*N)
# Infectious: 1 node
n_i_rand = initial_infecteds
n_s_rand = N - n_r_rand - n_i_rand
# Convert to percentages (sum to 100 exactly)
p_r_rand = round(n_r_rand / N * 100) # 75%
p_i_rand = round(n_i_rand / N * 100) # 0.01%, should be at least 1%
p_s_rand = 100 - p_r_rand - max(1,p_i_rand) # assign rest
if p_i_rand < 1:
    p_i_rand = 1  # ensure at least 1% infectious if allowed by instructions or expected
    p_s_rand = 100 - p_r_rand - p_i_rand

# (2) Targeted vaccination
frac_k10 = 1125 / N  # empirically given: 1125 degree-10 nodes
n_r_target = round(frac_k10 * N)
n_i_target = initial_infecteds
n_s_target = N - n_r_target - n_i_target
p_r_target = round(n_r_target / N * 100)
p_i_target = round(n_i_target / N * 100)
p_s_target = 100 - p_r_target - max(1,p_i_target)
if p_i_target < 1:
    p_i_target = 1
    p_s_target = 100 - p_r_target - p_i_target

# Prepare output
scenario_init_conditions = [
    {'S': p_s_rand, 'I': p_i_rand, 'R': p_r_rand},
    {'S': p_s_target, 'I': p_i_target, 'R': p_r_target}
]

# Set parameters (applies to both scenarios)
parameters = {'beta': 1, 'mu': 1}

# Textual descriptions
init_descs = [
    "Randomly vaccinate 75% of nodes, seed 1 infectious, rest susceptible (percentages only, rounded)",
    "Targeted vaccination: remove (vaccinate) all degree-10 nodes (11.25% of population), seed 1 infectious, rest susceptible (percentages only, rounded)"
]

# Return results to main workspace
output_vals = {
    "parameters": parameters,
    "initial_condition_desc": init_descs,
    "initial_conditions": scenario_init_conditions
}
output_vals