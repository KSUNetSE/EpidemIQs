
# Chain of Thought:
# 1. Analytical: Calculate the epidemic threshold (R0) and final size for homogeneous and heterogeneous networks using SEIR-like equations.
#    - For homogeneous: classic ODE and threshold beta/gamma > 1 (scaled for SEIR), or NGM approach.
#    - For network: use heterogeneous mean-field; basic reproduction number R0_net = beta * mu/(mu+gamma) * <k^2>/<k> / gamma (approximate for SEIR)
# 2. Use saved degree moments and typical parameters 
# Parameters chosen (COVID-like): beta = 0.25, mu = 1/3, gamma = 1/5
# Calculate and compare the threshold for both networks

def seir_thresholds(beta, mu, gamma, k1, k2):
    # For ODE (homogeneous): R0_h = beta/mu * mu/gamma = beta/gamma
    R0_h = beta / gamma
    # For heterogeneous net (pastor-satorras + SEIR): R0_net = beta * mu/(mu+gamma) * k2/k1 / gamma
    R0_net = (beta * mu / (mu + gamma)) * (k2 / k1) / gamma
    return R0_h, R0_net

beta = 0.25  # per edge per day
mu = 1/3     # incubation 3 days
gamma = 1/5  # infectious 5 days

# Degree moments from previous code
ER_k1 = 8.029
ER_k2 = 72.125
BA_k1 = 7.984
BA_k2 = 158.392

R0_h, R0_ER = seir_thresholds(beta, mu, gamma, ER_k1, ER_k2)
R0_h2, R0_BA = seir_thresholds(beta, mu, gamma, BA_k1, BA_k2)
results = {
    'Homogeneous_Mixing_R0': R0_h,
    'ER_Network_R0': R0_ER,
    'BA_Network_R0': R0_BA
}
results