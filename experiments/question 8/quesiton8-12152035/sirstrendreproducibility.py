
# Parameter and initial condition setup for SIRS (U-P-F) network model on Watts-Strogatz

# Mechanistic model compartments (fixed): U (Unaware), P (Posting), F (Fatigued)
# All rates are stochastic (hazard rates) for CTMC simulation

# Parameter sweep setup (covering broad but plausible range)
beta_range = [0.15, 0.3, 0.5]  # per-contact transmission rate (U-P), chosen to straddle moderate/strong transmission

gamma_range = [0.08, 0.15]      # fatigue rate (P->F), range covers typical durable but not too-fast fatigue

xi_range = [0.005, 0.015]    # forgetting rate (F->U), << gamma to ensure SIRS recurrency

parameter_grid = []
for beta in beta_range:
    for gamma in gamma_range:
        for xi in xi_range:
            parameter_grid.append({"beta": beta, "gamma": gamma, "xi": xi})

# Baseline scenario (mid-range)
baseline_parameters = {"beta": 0.3, "gamma": 0.1, "xi": 0.01}  # well within oscillatory regime based on literature

# Initial condition(s): 99% Unaware, 1% Posting, 0% Fatigued (reasonable classic contagion seed, N=10,000)
init_cond = [{"U": 99, "P": 1, "F": 0}]
init_cond_desc = ["Random selection: 99% Unaware, 1% Posting, 0% Fatigued (per routine viral trend seeding; no initial Fatigued)"]

# Output results
result = {
    "parameters": {"scan": parameter_grid, "baseline": baseline_parameters},
    "initial_condition_desc": init_cond_desc,
    "initial_conditions": init_cond
}
result