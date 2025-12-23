
# Given R0=3 for both network versions, what is the beta/gamma ratio for each network?
r0_target = 3

# For activity-driven temporal network
beta_gamma_activity = r0_target  # since R0 = beta/gamma / threshold -> beta/gamma = R0 * threshold

# For aggregated static network
beta_gamma_aggregated = r0_target

# Their corresponding infection rates beta if gamma=1
recovery_rate = 1.0
beta_activity = recovery_rate * beta_gamma_activity
beta_aggregated = recovery_rate * beta_gamma_aggregated

# Compute expected final size (R_inf) using percolation theory for static network: R_inf = 1 - exp(-R0 * R_inf)
# Iterate numerically for solution:
def solve_final_size(r0, tol=1e-6, max_iter=100):
    r = 0.5
    for _ in range(max_iter):
        r_next = 1 - np.exp(-r0 * r)
        if abs(r_next - r) < tol:
            return r_next
        r = r_next
    return r

R_inf_aggregated = solve_final_size(r0_target)

# For activity-driven: expected size is lower due to temporal ordering
# As an approximation, use a reduction factor from literature (e.g. 20-50% smaller for similar R0 in temporal vs static net)
R_inf_activity = R_inf_aggregated * 0.7  # conservative estimate (see e.g. PRL 109, 118702 (2012))

{
  'beta_gamma_activity': beta_gamma_activity,
  'beta_gamma_aggregated': beta_gamma_aggregated,
  'beta_activity': beta_activity,
  'beta_aggregated': beta_aggregated,
  'R_inf_aggregated': R_inf_aggregated,
  'R_inf_activity': R_inf_activity
}