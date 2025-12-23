
# Chain-of-Thought:
# Now, calculate SIR infection rate beta for static and temporal (activity-driven) settings, so that R0 = 3 in both. 
# Given R0 = beta/gamma * <k> in static, or, for activity-driven, R0 = (alpha * m * beta) / gamma (see Perra et al. 2012 Nat. Sci. Rep. and literature).
# We'll set gamma=0.2 for concrete rates (5-day infectious period, typical) and scale beta.
# Return both sets of rates.

mean_k_static = 329.636
mean_k2_static = 108941.0
alpha = 0.1
m = 2
R0 = 3
# Chosen infectious period (days): gamma = 0.2

params_out = {}
# (1) Static
# R0 = beta / gamma * <k> => beta_static = R0 * gamma / <k>
gamma = 0.2
beta_static = R0 * gamma / mean_k_static
params_out['beta_static'] = beta_static
params_out['gamma_static'] = gamma
# (2) Activity-driven:
# For Poisson activation SIR: R0 = (alpha * m * beta_temporal) / gamma
beta_temporal = R0 * gamma / (alpha * m)
params_out['beta_temporal'] = beta_temporal
params_out['gamma_temporal'] = gamma
params_out['static_desc'] = f'SIR params for static: beta={beta_static:.5f}, gamma={gamma} with mean_k={mean_k_static:.1f}'
params_out['temporal_desc'] = f'SIR params for activity-driven: beta={beta_temporal:.5f}, gamma={gamma} with alpha={alpha}, m={m}'
params_out