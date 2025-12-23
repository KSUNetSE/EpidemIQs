
# Parameters per step-by-step plan
N = 1000
R0 = 3.0
alpha = 0.1
m = 5
# (I) Activity-driven temporal network (per-contact rate for CTMC)
gamma_temporal = 1.0
beta_temporal = R0 * gamma_temporal / (m * alpha) # β = 3 / (5*0.1) = 6.0
# (II) Static aggregated network (ER-like projection, per-edge rate for CTMC; mean degree provided)
mean_k = 630.93
# Use R0 = (β/γ)⟨k⟩
gamma_static = 1.0  # Recovery rate is the same for mechanistic alignment
beta_static = R0 * gamma_static / mean_k # β ≈ 3 / 630.93 ≈ 0.00476
parameters = {
    "temporal_activity_driven": [beta_temporal, gamma_temporal],
    "static_aggregated_erlike": [beta_static, gamma_static]
}
initial_condition_desc = [
    "Randomly pick one node (out of 1000) to start infected (I=1); the remaining 999 are susceptible (S); recovered (R) count is zero (both temporal and static)."
]
initial_conditions = [{"S": 999, "I": 1, "R": 0}]
