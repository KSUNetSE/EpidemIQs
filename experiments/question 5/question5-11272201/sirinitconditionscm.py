
# Initial condition calculation for each vaccination arm
N = 10000 # network size (matches network description)

# Random vaccination arm
v_rand = 0.75 # 75% vaccinated randomly (R)
num_R_rand = int(round(N * v_rand))
num_remaining_rand = N - num_R_rand
num_I_rand = int(round(num_remaining_rand * 0.01)) # 1% of unvaccinated get infected
num_S_rand = num_remaining_rand - num_I_rand

# Convert to percentages
S_rand = int(round(100 * num_S_rand / N))
I_rand = int(round(100 * num_I_rand / N))
R_rand = int(round(100 * num_R_rand / N))
# Correction if rounding error
adj_rand = 100 - (S_rand + I_rand + R_rand)
S_rand += adj_rand

# Targeted vaccination (degree-10 nodes vaccinated, ~10% or as described in network)
f_target = 0.099 # 10% degree-10 nodes, targeted removal
num_R_target = int(round(N * f_target))
num_remaining_target = N - num_R_target
num_I_target = int(round(num_remaining_target * 0.01))
num_S_target = num_remaining_target - num_I_target

S_target = int(round(100 * num_S_target / N))
I_target = int(round(100 * num_I_target / N))
R_target = int(round(100 * num_R_target / N))
adj_target = 100 - (S_target + I_target + R_target)
S_target += adj_target

# Results: list of initial fraction dicts for each scenario
i_conditions = [
    {"S": S_rand, "I": I_rand, "R": R_rand},
    {"S": S_target, "I": I_target, "R": R_target}
]

i_conditions