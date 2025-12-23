
# Chain of Thought:
# 1. We aim to set up the SEIR model compartments and transitions for network-based (stochastic) simulation.
# 2. Initial condition: For a population of 2000, seed 10 initial infections, rest susceptible (no initial exposed or recovered).
# 3. Epidemic parameters based on literature for diseases with SEIR structure (e.g. COVID-like):
#    - Incubation period: 4 days (sigma = 1/4)
#    - Infectious period: 6 days (gamma = 1/6)
#    - R0: value chosen for comparative, e.g. 2.5
#    - We need to set infection rate beta for each network type, using network moments and R0.
# 4. For SEIR, R0 = beta / gamma * q, with q = (<k^2>-<k>)/<k>
# 5. We'll compute beta_ER and beta_BA accordingly.
# 6. Save these values for later use in stochastic simulation.

# Model definition and parameter calculation
N = 2000
R0 = 2.5
sigma = 1/4 # Exposed -> Infectious rate
gamma = 1/6 # Infectious -> Recovered rate

# Network statistics from previous step
ER_mean_k = 12.046
ER_k2 = 157.326
BA_mean_k = 11.964
BA_k2 = 315.544

q_ER = (ER_k2 - ER_mean_k) / ER_mean_k
q_BA = (BA_k2 - BA_mean_k) / BA_mean_k

# For network-based SEIR, beta = R0 * gamma / q
beta_ER = R0 * gamma / q_ER
beta_BA = R0 * gamma / q_BA

model_info = {
    'compartments': ['S', 'E', 'I', 'R'],
    'transitions': {
        'S-(I)->E': 'beta',
        'E->I': 'sigma',
        'I->R': 'gamma',
    },
    'params_ER': {'beta': beta_ER, 'sigma': sigma, 'gamma': gamma},
    'params_BA': {'beta': beta_BA, 'sigma': sigma, 'gamma': gamma},
    'initial_conditions': {'S': N-10, 'E': 0, 'I': 10, 'R': 0},
    'reasoning': 'Parameters derived for comparative SEIR on ER and BA networks. R0 fixed, infection rate set using degree moments per net.'
}
model_info
# Plan for analytical rates and parameters
# For SEIR, transitions:
# S -(I)-> E (rate beta), E->I (rate alpha), I->R (rate gamma)
# R_0 usually set for analysis.
# Analytical threshold: homogeneous, R_0 = beta/gamma; heterogeneous, R_0 = beta * <k^2>/<k> / gamma
# Let's set R_0=2.5 (illustrative COVID-like)

mean_k_hom = 10.0
mean_k2_hom = 100.0
mean_k_ba = 9.95
mean_k2_ba = 199.798

R0 = 2.5
# Recovery rate
gamma = 1/7  # infectious period ~7 days
# Incubation period 1/alpha ~ 3 days
alpha = 1/3

# Homogeneous: beta = R0 * gamma
beta_hom = R0 * gamma

# Heterogeneous: beta = R0*gamma * mean_k/(mean_k2)
beta_het = R0 * gamma * mean_k_ba / mean_k2_ba

params = {
    'homogeneous': {'beta': beta_hom, 'alpha': alpha, 'gamma': gamma},
    'heterogeneous': {'beta': beta_het, 'alpha': alpha, 'gamma': gamma}
}

params
