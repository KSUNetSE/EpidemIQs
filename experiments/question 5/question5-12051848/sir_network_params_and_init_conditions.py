
# Parameter and initial condition calculations for both network types and both vaccination strategies
import math

def get_percentage_counts(total, percentages):
    # Returns a list of counts rounded so sum == total
    floats = [p/100*total for p in percentages]
    ints = [int(round(f)) for f in floats]
    delta = total - sum(ints)
    # Correction: add delta to S if needed
    if delta != 0:
        for i in range(len(ints)):
            if (percentages[i] == max(percentages)) and (delta != 0):
                ints[i] += delta
                break
    return ints

# Given parameters
N = 10000
R0 = 4.0
network_stats_poisson = {'mean_k': 2.995, 'mean_k2': 11.93, 'q': 2.99}
network_stats_tailored = {'mean_k': 2.998, 'mean_k2': 14.97, 'q': 3.99}
vacc_random_pc = 0.75 # 75% random vaccination
# Tailored network (k=2,3,10): degree 10 fraction: ~10.6%
frac_degree10_tailored = 0.106
frac_degree10_poisson = 0.0014

# --- Step 1: Transmission Probabilities
# For percolation, T = R0/q
T_poisson = R0/network_stats_poisson['q']
T_tailored = R0/network_stats_tailored['q']
# For dynamic CTMC: Need beta_edge and gamma with T = beta/(beta+gamma)
# Use gamma = 1 (unit timescale)
def beta_from_T(T, gamma):
    if T >= 1.0:
        return float('inf') # Perfect transmission, mathematically infinite rate
    return gamma * T / (1-T)

gamma = 1.0
# For this scenario, T ~= 1 gives beta_edge >> gamma
# We'll note this, but for code stability in simulation, use T=0.99 as well
T_sims = [1.0, 0.99]  # present both options
beta_edge_poisson = [beta_from_T(T, gamma) for T in T_sims]
beta_edge_tailored = [beta_from_T(T, gamma) for T in T_sims]

# --- Step 2: Initial Conditions Percentages ---
# (a) Poisson network, random vaccination (pc = 75%)
p_vacc_rand = vacc_random_pc
I0_pct = 1 # Seed 1% infected (unless scenario mandates only 1 person, but for percent, 1%)
R_pct = int(round(100*p_vacc_rand))
S_pct = 100 - R_pct - I0_pct
init_poisson_random_vacc = {'S': S_pct, 'I': I0_pct, 'R': R_pct}

# (b) Tailored network, targeted vacc k=10 (10.6% of nodes)
p_vacc_targeted = frac_degree10_tailored
R_pct_targeted = int(round(100*p_vacc_targeted))
I0_pct = 1
S_pct_targeted = 100 - R_pct_targeted - I0_pct
init_tailored_targeted_vacc = {'S': S_pct_targeted, 'I': I0_pct, 'R': R_pct_targeted}

# (c) Poisson network, targeted vacc k=10 (only 0.14% degree-10)
p_vacc_targeted_pois = frac_degree10_poisson
R_pct_targeted_pois = int(round(100*p_vacc_targeted_pois))
I0_pct = 1
S_pct_targeted_pois = 100 - R_pct_targeted_pois - I0_pct
init_poisson_targeted_vacc = {'S': S_pct_targeted_pois, 'I': I0_pct, 'R': R_pct_targeted_pois}

# (d) Tailored network, random vaccination (75%)
R_pct_rand_tail = int(round(100*vacc_random_pc))
S_pct_rand_tail = 100 - R_pct_rand_tail - I0_pct
init_tailored_random_vacc = {'S': S_pct_rand_tail, 'I': I0_pct, 'R': R_pct_rand_tail}

# Record initial condition descriptions
initial_condition_desc = [
    'Random vaccination (75%) in Poisson(3) network. Initial infection seeded randomly (1%)',
    'Targeted vaccination (all k=10) in tailored (k=2,3,10, ~10.6% degree-10) network. Initial infection seeded randomly (1%)',
    'Targeted vaccination (all k=10) in Poisson(3) network. Initial infection seeded randomly (1%)',
    'Random vaccination (75%) in tailored (k=2,3,10) network. Initial infection seeded randomly (1%)'
]

# Organize parameters for both simulation types (dynamic rates and percolation probabilities)
parameters = {
    'poisson_random': {
        'T (probability)': T_poisson,
        'beta_edge (rate)': beta_edge_poisson[0],
        'beta_edge (rate, T=0.99)': beta_edge_poisson[1],
        'gamma (rate)': gamma
    },
    'tailored_targeted': {
        'T (probability)': T_tailored,
        'beta_edge (rate)': beta_edge_tailored[0],
        'beta_edge (rate, T=0.99)': beta_edge_tailored[1],
        'gamma (rate)': gamma
    },
    'poisson_targeted': {
        'T (probability)': T_poisson,
        'beta_edge (rate)': beta_edge_poisson[0],
        'beta_edge (rate, T=0.99)': beta_edge_poisson[1],
        'gamma (rate)': gamma
    },
    'tailored_random': {
        'T (probability)': T_tailored,
        'beta_edge (rate)': beta_edge_tailored[0],
        'beta_edge (rate, T=0.99)': beta_edge_tailored[1],
        'gamma (rate)': gamma
    },
}
# Initial condition list
initial_conditions = [init_poisson_random_vacc, init_tailored_targeted_vacc, init_poisson_targeted_vacc, init_tailored_random_vacc]

return_vars = ['parameters', 'initial_condition_desc', 'initial_conditions', 'T_poisson', 'T_tailored', 'beta_edge_poisson', 'beta_edge_tailored','gamma']
