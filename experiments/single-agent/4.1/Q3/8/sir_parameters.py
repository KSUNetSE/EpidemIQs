
# Find SIR model infection and recovery rates for R0=3 on
# a) static-aggregated network
# b) temporal (expected network stats)

N = 1000
R0 = 3

# For both, assume homogeneous infection/recovery parameters
recovery_rate = 1.0/5  # infectious period = 5 time units (arbitrary, similar to many SIR studies)

def infection_rate_from_R0(R0, gamma, k_mean, k2_mean):
    # Mean excess degree q = (<k^2> - <k>) / <k>
    q = (k2_mean - k_mean) / k_mean if k_mean > 0 else 0
    if q != 0:
        beta = R0 * gamma / q
    else:
        beta = 0.0
    return beta, q

# a) For static-aggregated network
k_mean_static = 38.99
k2_mean_static = 1570.474
beta_static, q_static = infection_rate_from_R0(R0, recovery_rate, k_mean_static, k2_mean_static)
# b) For time-averaged temporal network
k_mean_temp = 0.398
k2_mean_temp = 0.7158
beta_temp, q_temp = infection_rate_from_R0(R0, recovery_rate, k_mean_temp, k2_mean_temp)

return_vars = {
    'beta_static': beta_static,
    'q_static': q_static,
    'beta_temp': beta_temp,
    'q_temp': q_temp,
    'recovery_rate': recovery_rate
}
