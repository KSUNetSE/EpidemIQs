
# Mechanistic parameter and initial condition assignment for SIRS-UPF on Watts–Strogatz network
parameters = {
    'beta': 0.8,   # edge-based hazard rate from Posting (P) neighbor to Unaware (U) [per day]
    'gamma': 0.3,  # node-based hazard rate for Posting (P) -> Fatigued (F) [per day]
    'xi': 0.1      # node-based hazard rate for Fatigued (F) -> Unaware (U) [per day]
}
# Initial condition: 1% Posting, 99% Unaware, 0% Fatigued, for N = 1000
N = 1000
initial_pct = {'U': round(0.99 * 100), 'P': round(0.01 * 100), 'F': 0}
# Diagnostics: implied network reproduction number (locally tree-like approx)
k = 10 # mean degree
T = parameters['beta'] / (parameters['beta'] + parameters['gamma'])
R0_network = k * T # not used for tuning; diagnostic only

return_vars = ['parameters', 'initial_pct', 'k', 'T', 'R0_network']
