
# SIR parameter and initial condition calculation for scenario

def get_sir_discrete_params_and_initials(N, R0, gamma, m, alpha, seeding_modes=['random', 'targeted']):
    # Given in scenario: For activity-driven, R0 = (beta/gamma) * (2m*alpha) => beta/gamma = R0 / (2m*alpha)
    # Given: N=1000, R0=3, gamma=1, m=2, alpha=0.1
    #      beta/gamma = 7.5 -> beta = 7.5 * gamma = 7.5
    beta = 7.5
    # Both models use same beta and gamma (discrete-time probabilities, not rates)
    params = {
        'temporal_activity_driven': {"beta": beta, "gamma": gamma},
        'static_aggregated': {"beta": beta, "gamma": gamma},
    }
    # Initial conditions: Single infected, rest susceptible, none recovered
    I = 1
    S = N - I
    R = 0
    # As percentages:
    total = S + I + R
    S_pct = round(100 * S / total)
    I_pct = round(100 * I / total)
    R_pct = round(100 * R / total)
    # After rounding, sum may not be exactly 100; adjust S/R (make sure I_pct is 1)
    if (S_pct + I_pct + R_pct) != 100:
        # Keep I_pct=1 (minimum seeding), credit difference to S
        delta = 100 - (S_pct + I_pct + R_pct)
        S_pct += delta
    # For optional: if seeding at high-activity (or high-degree) node
    initials_list = []
    I_types = []
    for mode in seeding_modes:
        if mode == 'random':
            I_types.append("Random single infected node, rest susceptible (standard scenario)")
            initials_list.append({'S': S_pct, 'I': I_pct, 'R': R_pct})
        elif mode == 'targeted':
            I_types.append("Targeted seed at high-activity (or high-degree) node, rest susceptible (optional scenario)")
            initials_list.append({'S': S_pct, 'I': I_pct, 'R': R_pct})
    return params, I_types, initials_list

# Call for this scenario
params, I_types, initials_list = get_sir_discrete_params_and_initials(N=1000, R0=3.0, gamma=1.0, m=2, alpha=0.1, seeding_modes=['random', 'targeted'])

# Adjust initial seeding percentage to ensure at least 1% initial infection, per guideline (avoid all 0 I)
# For N=1000, 1 infected = 0.1%, but simulation guidance: initial infection must not be 0.
def min_infected_initial_condition(N):
    # Enforce at least 1% initial seed if possible
    min_pct = max(round(100 / N), 1)  # at least 1% or nearest integer
    S_pct = 100 - min_pct
    R_pct = 0
    return {'S': S_pct, 'I': min_pct, 'R': R_pct}

# For N=1000, this will be 1% infected (I=10), which is reasonable for simulation study reproducibility
min_infected = min_infected_initial_condition(1000)
# Two cases: random and targeted seeding as before, but using new convention
initials_list = [min_infected, min_infected]
