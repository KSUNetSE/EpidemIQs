
# Step 1: Input the extracted network statistics
k_mean = 46.8
k2_mean = 2233.5
D = (k2_mean - k_mean)/k_mean

# Step 2: SEIQR Model: parameter regime and empirical values
R0 = 1.66
incubation_days = 1 # days (mean duration E)
hidden_infectious_days = 0.9 # days (mean duration I)
quarantine_recovery_days = 2 # days (mean duration Q)

sigma = 1 / incubation_days  # E->I rate
# tau: Hidden infectious to quarantine
tau = 1 / hidden_infectious_days
# gamma: Quarantine to recovery
gamma = 1 / quarantine_recovery_days

# Step 3: Compute beta for required R0 on network by heterogeneous mean-field:
# R0 = (beta/tau) * D  =>  beta = R0 * tau / D
beta = R0 * tau / D

# Step 4: Initial conditions, population N=763: S=762, E=0,I=1,Q=0,R=0.
N = 763
init_S = 762
init_E = 0
init_I = 1
init_Q = 0
init_R = 0
# Express as integer percentages summing to 100
S_pct = round(100 * init_S / N)
E_pct = round(100 * init_E / N)
I_pct = round(100 * init_I / N)
Q_pct = round(100 * init_Q / N)
R_pct = round(100 * init_R / N)
# Correct to ensure total is exactly 100
init_pcts = [S_pct, E_pct, I_pct, Q_pct, R_pct]
diff = 100 - sum(init_pcts)
# Always adjust S downward or upward since it's largest
init_pcts[0] += diff
initial_condition_dict = dict(zip(["S","E","I","Q","R"], init_pcts))

parameters = {"beta": beta, "sigma": sigma, "tau": tau, "gamma": gamma}

# Save all numerical results for output
dict_out = {
    "beta": beta,
    "sigma": sigma,
    "tau": tau,
    "gamma": gamma,
    "initial_condition_percentages": initial_condition_dict
}

dict_out
