
# Parameter setup for SIRV (SIR + Vaccination-as-removal) mechanistic model on uncorrelated config-model network

# Given network parameters:
mean_excess_degree_theory = 4.0  # for analytic match
mean_excess_degree_empirical = 4.24  # from network construction
R0 = 4.0  # given intrinsic basic reproduction number
T = 1.0   # per-edge transmissibility (prob. of infection if edge exists)

# For CTMC on static network: R0 = (beta/gamma) * q
q_use = mean_excess_degree_theory  # Use theoretical q for direct analytic consistency
# solving for beta/gamma:
beta_over_gamma = R0 / q_use  # = 1.0

# Set gamma (recovery/removal rate); infectious period = 1/gamma.
gamma = 1.0  # mean infectious period = 1 time unit (arbitrary since T=1)
beta = beta_over_gamma * gamma  # = 1.0

# For validation runs, these can be swept or changed, but default is 1.0, 1.0
parameters = {"beta": beta, "gamma": gamma}

# --- Initial Conditions ---
# Population size
N = 10000
# Cases to specify:
# 1. Random Vaccination, for v in [0.6, 0.75, 0.8, 0.9] (simulate sweep, but show a few)
r_vacs = [0.6, 0.75, 0.8, 0.9]
random_init_conditions = []
for v in r_vacs:
    num_r = round(N * v)
    num_i = 1
    num_s = N - num_r - num_i
    # Express as percentage, rounded to integers summing to 100
    S_pct = round(100 * num_s / N)
    I_pct = round(100 * num_i / N)
    R_pct = 100 - S_pct - I_pct
    random_init_conditions.append({"S": S_pct, "I": I_pct, "R": R_pct})

# 2. Targeted Vaccination on degree-10 nodes
p10 = 0.119  # fraction of degree-10 nodes in population, empirical network
# Formula-derived herd immunity threshold: v_c = 0.1125 = 11.25% of TOTAL pop.
v_c_targeted = 9/80  # theoretical
f_min = v_c_targeted / p10  # minimum fraction of degree-10 nodes to vaccinate to hit analytic threshold
# Simulate a few relevant targeted vaccination scenarios:
target_fs = [0.0, f_min, 1.0]  # None, threshold/analytic, or all degree-10s (upper bound)
targeted_init_conditions = []
for f in target_fs:
    vac_degree10 = round(N * p10 * f)
    num_i = 1
    num_r = vac_degree10
    num_s = N - num_r - num_i
    # Express as percentage, rounded, sum to 100
    S_pct = round(100 * num_s / N)
    I_pct = round(100 * num_i / N)
    R_pct = 100 - S_pct - I_pct
    targeted_init_conditions.append({"S": S_pct, "I": I_pct, "R": R_pct})

# Descriptions
random_desc = [f"Randomly vaccinate {int(v*100)}% of nodes, seed with 1 case" for v in r_vacs]
targeted_desc = [
    "No vaccination (all degree-10 nodes susceptible)",
    f"Vaccinate just enough (~{f_min*100:.1f}%) of degree-10 nodes to hit theoretical threshold (covers {v_c_targeted*100:.1f}% of population)",
    f"Vaccinate ALL degree-10 nodes (covers {round(100*p10)}% of population)"
]

# Collate
initial_condition_types = random_desc + targeted_desc
initial_conditions = random_init_conditions + targeted_init_conditions

