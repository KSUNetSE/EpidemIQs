
# Reconstruct formulas and values for beta calibration, test each part stepwise
import math

R0 = 3.0
mean_inf_period = 5.0
N = 1000
init_I = 5
init_S = N - init_I
init_R = 0
alpha = 0.1
m = 2

gamma = 1/mean_inf_period
# For temporal activity-driven network
beta_temporal = (R0 * gamma) / (2 * m * alpha)
# For static (try fallback formula: use well-mixed/ER-case as beta = (R0 * gamma)/<k>, with <k>=2 )
k_avg = 2
beta_static_fallback = (R0 * gamma) / k_avg

# For reporting: per-edge transmissibility for static (if using above beta):
T_edge_static = 1 - math.exp(-k_avg * beta_static_fallback / gamma)

# Compute initial conditions in percent
frac_S = round((init_S / N) * 100)
frac_I = round((init_I / N) * 100)
frac_R = round((init_R / N) * 100)
if (frac_S + frac_I + frac_R) != 100:
    frac_S = 100 - (frac_I + frac_R)
initial_conditions = [{"S": frac_S, "I": frac_I, "R": frac_R}]

results = {
    "pars_temporal": {"beta": beta_temporal, "gamma": gamma},
    "pars_static": {"beta": beta_static_fallback, "gamma": gamma},
    "T_edge_static": T_edge_static,
    "initial_conditions": initial_conditions
}
results