
# Competitive multiplex SIS (mutually exclusive) CTMC parameter setting and initial condition construction
import numpy as np

# Step 1: Network and threshold parameters
eig_A = 19.14   # largest eigenvalue of A
eig_B = 16.87   # largest eigenvalue of B
N = 1000        # network size

# Step 2: Choose several (tau1, tau2) beyond critical for both coexistence and dominance checks
# Critical values (just at threshold): tau1_c = 1/eig_A, tau2_c = 1/eig_B
tau1_crit = 1 / eig_A
# 0.05226
tau2_crit = 1 / eig_B
# 0.05929

# Choose several multipliers c for supercritical:
tau_multipliers = [1.1, 1.3, 1.7, 2.0]

# We'll test two sets of recovery rates for interpretability
delta1_cases = [1.0, 0.5]
delta2_cases = [1.0, 0.5]

parameters = {}
case_id = 0
for delta1 in delta1_cases:
    for delta2 in delta2_cases:
        for c1 in tau_multipliers:
            for c2 in tau_multipliers:
                tau1 = c1 * tau1_crit
                tau2 = c2 * tau2_crit
                beta1 = tau1 * delta1
                beta2 = tau2 * delta2
                case_descr = f"delta1={delta1}, delta2={delta2}; tau1~{round(tau1,3)} (c1={c1}), tau2~{round(tau2,3)} (c2={c2})"
                key = f"case_{case_id}: {case_descr}"
                parameters[key] = {
                    'beta1': round(beta1, 5),
                    'delta1': round(delta1, 5),
                    'beta2': round(beta2, 5),
                    'delta2': round(delta2, 5),
                    'tau1': round(tau1, 5),
                    'tau2': round(tau2, 5),
                }
                case_id += 1

# Step 3: Initial conditions construction
# Description:
initial_condition_desc = [
    "Random 1% of nodes infected with virus 1, rest susceptible",  # I1-seed
    "Random 1% of nodes infected with virus 2, rest susceptible",  # I2-seed
    "Random 1% of nodes infected with each virus, rest susceptible" # Both-seed (for strong coexistence regime)
]

# Express as percentages (integer values, total=100)
def make_initial(S, I1, I2):
    # Force integer and sum to 100
    vals = np.array([S, I1, I2])
    vals = np.round(vals/np.sum(vals)*100).astype(int)
    # Correct rounding so sum exactly 100
    delta = 100 - np.sum(vals)
    if delta != 0:
        idx = np.argmax(vals) if delta < 0 else np.argmin(vals)
        vals[idx] += delta
    return {'S': int(vals[0]), 'I1': int(vals[1]), 'I2': int(vals[2])}

initial_conditions = [
    make_initial(99, 1, 0),
    make_initial(99, 0, 1),
    make_initial(98, 1, 1)
]

parameters, initial_condition_desc, initial_conditions