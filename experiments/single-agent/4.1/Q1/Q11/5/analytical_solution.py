
# --- Analytical calculation for deterministic SEIR on homogeneous and heterogeneous network
# R0 for homogeneous network: R0 = beta/gamma * <k>
# For heterogeneous: R0 = beta/gamma * (<k^2> - <k>)/<k>
# We'll illustrate the difference for matched mean degree.

# Assume plausible SEIR rates for a respiratory infection (e.g., COVID, flu):
# Latent period = 3 days (sigma=1/3)
# Infectious period = 5 days (gamma=1/5)
# Target R0 in homogeneous-mixing: ~2.5
latent_period = 3  # days
infectious_period = 5  # days
sigma = 1/latent_period
gamma = 1/infectious_period
R0_hom = 2.5
mean_k_er = 12.15
second_k_er = 158.54
mean_k_ba = 11.93
second_k_ba = 283.33

# Analytical beta values
beta_er = R0_hom * gamma / mean_k_er
beta_ba = R0_hom * gamma * mean_k_er / (second_k_ba - mean_k_ba)

# Analytical final epidemic size approximation for deterministic model
# For SIR, can use final size equation. For SEIR, outbreak size is similar if duration long/large N, so we can compare beta/gamma ratio and R0 analytically.

analytical_summary = {
    'SEIR_parameters': {
        'sigma': sigma,
        'gamma': gamma,
        'target_R0': R0_hom,
        'latent_period': latent_period,
        'infectious_period': infectious_period
    },
    'network_stats': {
        'er': {'mean_k': mean_k_er, 'second_k': second_k_er, 'beta': beta_er},
        'ba': {'mean_k': mean_k_ba, 'second_k': second_k_ba, 'beta': beta_ba}
    },
    'R0_formulas': {
        'homogeneous_mixing': 'R0 = beta/gamma * <k>',
        'heterogeneous_network': 'R0 = beta/gamma * (<k^2> - <k>)/<k>'
    },
    'beta_selected': {'beta_er': beta_er, 'beta_ba': beta_ba}
}

analytical_summary