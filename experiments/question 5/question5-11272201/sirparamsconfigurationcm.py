
import numpy as np

def infer_sir_params(network_stats, R0, gamma=1.0):
    # network_stats: dict with keys 'mean_deg' <k>, 'sec_moment' <k^2>
    k_mean = network_stats['mean_deg']
    k2_mean = network_stats['sec_moment']
    D = (k2_mean - k_mean) / k_mean
    T = R0 / D  # desired per-edge transmissibility (dimensionless)
    beta = (T * gamma) / (1 - T) if T < 1.0 else 1e9  # set to large if T==1 (no fadeout, non-biological)
    return {
        "beta": beta,  # per-edge infection rate (CTMC)
        "gamma": gamma,  # recovery rate (CTMC)
        "T": T,  # transmissibility (for reference here)
        "D": D  # for checking math
    }

# -- Scenario values --
network_stats = {'mean_deg': 2.99, 'sec_moment': 14.95}  # matches provided network
R0 = 4.0
recovery_rate = 1.0 # day^-1 (biologically plausible time unit)
params = infer_sir_params(network_stats, R0, gamma=recovery_rate)

params