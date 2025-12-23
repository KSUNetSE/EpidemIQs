
import numpy as np

# Step 1: Inputs (from network and epidemiological summary)
R0 = 8.0  # intrinsic R0 from scenario
n = 763
mean_k = 68.52  # <k>
mean_k2 = 5077.23  # <k^2>

# Mean generation time (latent + infectious, i.e., 1/sigma + 1/gamma)
gen_time = 1.9  # days

# Typical clinical residence times for B and C stages (influenza, from literature)
mean_B = 2.5  # days in B (confined to bed)
mean_C = 3.0  # days in C (convalescent)

# Step 2: Solve for sigma and gamma, given mean gen time constraint
# Let x = 1/sigma, y = 1/gamma, x + y = gen_time. If we split as (latent,infectious) = (1,0.9 days),
latent = 1.0  # 1/sigma
infectious = gen_time - latent  # 0.9 days
sigma = 1/latent

gamma = 1/infectious

# Step 3: Compute network denominator for heterogeneous mean-field R0
# network denominator: D = (<k^2> - <k>)/<k>
D = (mean_k2 - mean_k) / mean_k

# Step 4: Relate R0 to beta:
# R0 ≈ T * D,   T = beta/(beta+gamma);  R0 = (beta/(beta+gamma)) * D
# Rearranged:   beta = (R0 * gamma) / (D - R0)
beta = (R0 * gamma) / (D - R0)

# Step 5: Set delta, kappa from B, C durations.
delta = 1/mean_B
kappa = 1/mean_C

# Step 6: Initial conditions as percentages (3 initial infections, rest susceptible). No E, B, C, R cases.
I0 = 3
S0 = n - I0
E0,B0,C0,R0 = 0,0,0,0

init = [S0, E0, I0, B0, C0, R0]
perc = [int(round(i * 100 / n)) for i in init]
# ensure sum is 100 (adjust S downwards or I upwards if rounding error)
diff = 100 - sum(perc)
if diff != 0:
    perc[0] += diff  # Adjust S (Susceptible) to make sum 100


params = {
    'beta': beta,
    'sigma': sigma,
    'gamma': gamma,
    'delta': delta,
    'kappa': kappa,
}

init_perc = {
    'S': perc[0],
    'E': perc[1],
    'I': perc[2],
    'B': perc[3],
    'C': perc[4],
    'R': perc[5],
}

output = {
    'parameters': params,
    'initial_conditions': [init_perc],
    'initial_condition_type': ['Seeding: 3 initial infectious nodes (all in I) in one house, remainder susceptible']
}
output