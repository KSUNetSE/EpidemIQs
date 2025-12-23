
# Parameter inference for SIR on temporal (activity-driven) and static weighted networks (Experiment 3)
N = 10000
mean_a = 0.0963         # inferred activity (per time step from diagnostics, ≈ <a> ~ 0.1)
mean_a2 = 0.0230        # <a^2>
m = 2                   # contacts per activation
gamma = 0.1             # per-node recovery rate
R0 = 3.0

# === Temporal (activity-driven) ===
# Use classic mean-field: R0 = m beta/gamma <a> -> beta = R0 * gamma / (m * <a>)
beta_temporal = R0 * gamma / (m * mean_a)

# Optional: heterogeneity-aware formula (not required by this scenario, but shown for completeness)
beta_temporal_hetero = R0 * gamma / (m * (mean_a + (mean_a2 ** 0.5)))

# === Static network ===
# Mean degree (per timestep, doubles mean_a*m due to undirected counting, but static: edges/duration N = 10,000)
mean_degree = 721.9     # from diagnostics

# For the static weighted network, per-edge transmission is beta*w_ij, where weights are normalized such that sum_j w_ij ≈ mean_degree
# Let's match mean β·⟨k⟩/gamma = R0. So: β = R0*gamma/⟨k⟩
beta_static = R0 * gamma / mean_degree

rates = {
    'temporal': [round(beta_temporal, 5), gamma],
    'temporal_hetero': [round(beta_temporal_hetero, 5), gamma],
    'static': [round(beta_static, 5), gamma]
}
rates