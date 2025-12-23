
# parametersetting_CTMC_SIR_vacc.py
# SIR (network, CTMC), T=1, vaccination strategies
# Network: config model N=10000, degree-dist: 0.75@2, ~0.143@3, ~0.106@10

import numpy as np

def params_and_initial_conditions():
    N = 10000
    T = 1   # Per-edge transmission probability (used as per scenario: SIR with T=1)
    gamma = 1  # Recovery rate
    # Since T=1, "infinite" beta makes T=1 for edge. But we use T directly.
    # Vaccination scenarios
    # Random vaccination thresholds to test (below, at, above critical)
    v_rand_cases = [0.5, 0.7, 0.75, 0.8]
    # Targeted vaccination: all degree-10 nodes (fraction ~0.106), or exactly 9/80 (0.1125) if available
    p_10 = 0.106
    v_target_fraction = min(p_10, 9/80) # Fraction of network possible to immunize by removing all k=10 nodes
    # Initial infection: 1% of S
    initial_infected_fraction = 0.01
    # Each scenario, compute initial condition (percentages, rounding)
    def ic_post_vacc(v_frac):
        # Susc left: N*(1-v_frac)
        n_S = N * (1-v_frac)
        n_I = int(round(n_S * initial_infected_fraction))
        n_S = int(round(n_S - n_I))
        n_R = int(round(N*v_frac))
        remainder = N - n_S - n_I - n_R
        n_S += remainder # adjust to ensure sum to N
        return {'S': int(round(100*n_S/N)), 'I': int(round(100*n_I/N)), 'R': int(round(100*n_R/N))}
    
    cases = {}
    ic_descs = []
    ics = []
    # Random vaccination cases
    for v in v_rand_cases:
        params = {"T": T, "gamma": gamma, "vaccinated_fraction": v}
        cases[f"Random v={v}"] = params
        ic_descs.append(f"Random vaccination; {int(100*v)}% randomly assigned immune (R) before epidemic.")
        ics.append(ic_post_vacc(v))
    # Targeted vaccination case
    params = {"T": T, "gamma": gamma, "vaccinated_fraction": p_10}
    cases[f"Targeted k=10 (all)"] = params
    ic_descs.append(f"Targeted vaccination; all degree-10 nodes (~10.6%) assigned immune (R) before epidemic.")
    ics.append(ic_post_vacc(p_10))
    return cases, ic_descs, ics

results = params_and_initial_conditions()
cases, ic_descs, ics = results
