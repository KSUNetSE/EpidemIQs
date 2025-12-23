
import numpy as np

# Given eigenvalues for two cases (lambda_1)
lambda1_A_high = 17.33
lambda1_B_high = 17.15
lambda1_A_low = 17.07
lambda1_B_low = 17.77

# Fix recovery rates according to best practice (delta1 = delta2 = 1)
delta1 = 1.0
delta2 = 1.0

# Define persistence/invasion thresholds for single virus
thresh_tau1_high = 1 / lambda1_A_high  # just at threshold
thresh_tau2_high = 1 / lambda1_B_high
thresh_tau1_low  = 1 / lambda1_A_low
thresh_tau2_low  = 1 / lambda1_B_low

# Choose effective rates for tau1, tau2 for different regimes:
# (1) just below threshold, (2) just above, (3) well above, (4) much above
# We match to allow scan of dominance, coexistence, bistability (the final choice will select 2 relevant cases for each network scenario)
tau_scenarios = [
    (thresh_tau1_high * 0.8, thresh_tau2_high * 0.8),    # both below (should disappear)
    (thresh_tau1_high * 1.05, thresh_tau2_high * 0.95),   # only tau1 above (V1 can invade)
    (thresh_tau1_high * 0.95, thresh_tau2_high * 1.05),   # only tau2 above (V2 can invade)
    (thresh_tau1_high * 1.20, thresh_tau2_high * 1.20),   # both well above (possible coexistence)
    (thresh_tau1_high * 1.5, thresh_tau2_high * 1.1),     # tau1 much above, tau2 weakly above (V1 dominance in high overlap)
    (thresh_tau1_high * 1.1, thresh_tau2_high * 1.5),     # tau2 much above, tau1 weakly above (V2 dominance in high overlap)
]

param_table_high = []
param_table_low = []

for tau1, tau2 in tau_scenarios:
    # high correlation
    data_high = {
        'beta1': tau1 * delta1,
        'delta1': delta1,
        'beta2': tau2 * delta2,
        'delta2': delta2,
        'tau1': tau1,
        'tau2': tau2,
    }
    # low correlation (swap only the eigenvalues for threshold as effective rates)
    tau1_low = tau1 * (thresh_tau1_low / thresh_tau1_high)
    tau2_low = tau2 * (thresh_tau2_low / thresh_tau2_high)
    data_low = {
        'beta1': tau1_low * delta1,
        'delta1': delta1,
        'beta2': tau2_low * delta2,
        'delta2': delta2,
        'tau1': tau1_low,
        'tau2': tau2_low,
    }
    param_table_high.append(data_high)
    param_table_low.append(data_low)

# Initial condition prescription
init_cond = {'S': 90, 'I1': 5, 'I2': 5}  # as per scenario (percentages, integers, sum to 100)

results = {
    'param_table_high': param_table_high,
    'param_table_low': param_table_low,
    'initial_conditions': [init_cond],
    'initial_condition_desc': [
        'Randomly assign 5% of nodes to I1 (Virus 1), 5% to I2 (Virus 2), remainder 90% susceptible; I1/I2 sets disjoint.'
    ]
}
