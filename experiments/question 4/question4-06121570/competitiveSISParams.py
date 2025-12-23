
# Step 1: Collect network stats
# Information for each layer and pairing:
# - mean degree <k>
# - second moment <k^2>
# - spectral radius (lambda1) if available (can estimate for large ER: lambda1~<k>, for BA: lambda1~sqrt(k_max), empirical <k^2>/<k> when possible)

# We have <k> and <k^2> for all layers and scenarios
# For illustration: maximally correlated ER multiplex
# LayerA: <k>=8.04, <k^2>=72.48
# LayerB: <k>=8.27, <k^2>=76.94 (uncorrelated)
# BA (corr): <k>=7.97, <k^2>=141.83
# BA (anticorr): <k>=7.97, <k^2>=135.10
# SBM LayerA: <k>=25.24, <k^2>=659.76
# SBM LayerB (high overlap): <k>=25.21, <k^2>=656.89; (partial): <k>=24.80, <k^2>=637.07
# We'll use q = (<k^2>-<k>)/<k> for the SIS threshold computation

def estimate_lambda1_er(k):
    # empirical for ER, lambda1 ~ <k>
    return k

def estimate_lambda1_sf(k, k2):
    # For large n, spectral radius in BA ~ sqrt(k_max). But use <k^2>/<k> as crude mean-field proxy
    return k2/k

def param_for_sis(q, delta, tau_factor=1.1):
    # set tau just above threshold: tau = 1.1, 1.05, etc.
    tau = tau_factor  # tau = beta/delta = (desired threshold)*factor
    beta = tau * delta / q
    return beta

params = {}

networks = {
    'ER_corr': {'k': 8.04, 'k2': 72.48, 'tau1_fac': 1.10, 'tau2_fac': 1.05},
    'ER_uncorr': {'k': 8.27, 'k2': 76.94, 'tau1_fac': 1.07, 'tau2_fac': 1.09},
    'BA_corr': {'k': 7.97, 'k2': 141.83, 'tau1_fac': 1.10, 'tau2_fac': 1.08},
    'BA_anticorr': {'k': 7.97, 'k2': 135.10, 'tau1_fac': 1.11, 'tau2_fac': 1.09},
    'SBM_highoverlap': {'k': 25.21, 'k2': 656.89, 'tau1_fac': 1.08, 'tau2_fac': 1.07},
    'SBM_partial': {'k': 24.80, 'k2': 637.07, 'tau1_fac': 1.11, 'tau2_fac': 1.12}
}

# Fix the recovery rate for all scenarios for comparability
delta1 = delta2 = 0.25 # Typical SIS (recovery per ~4 days)

for net, info in networks.items():
    k, k2 = info['k'], info['k2']
    q = (k2 - k) / k
    # for competitive model: LayerA (virus1), LayerB (virus2)
    beta1 = param_for_sis(q, delta1, tau_factor=info['tau1_fac'])
    beta2 = param_for_sis(q, delta2, tau_factor=info['tau2_fac'])
    params[net] = {'beta1': round(beta1, 5), 'delta1': delta1, 'beta2': round(beta2, 5), 'delta2': delta2, 'q': round(q, 2)}

# Initial conditions: N=1000. 2% for I1, 2% for I2, rest S
N = 1000
I1 = I2 = round(0.02 * N) # 20
S = N - I1 - I2
initial_conditions = [{'S': S, 'I1': I1, 'I2': I2}]
initial_condition_desc = ["Randomly assign 2% of nodes to I1, 2% to I2, rest susceptible; no co-infection"]

params, initial_conditions, initial_condition_desc