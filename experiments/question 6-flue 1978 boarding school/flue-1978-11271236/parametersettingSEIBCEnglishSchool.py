
# ====================
# parametersettingSEIBC_english_school.py
# Compute plausible rates for the S→E→I→B→C mechanistic model (CTMC rates), for static complete contact graph (K_763)
#
# --- 1. Set up scenario constants ---
N = 763
R0 = 1.65
# Literature/biological (influenza) mean durations (days):
latent_period_choices = [1, 2]       # plausible range (days)
infectious_period_choices = [1, 2]   # plausible range in I (days); pre-bed symptomatic, still able to infect
bed_period_choices = [2, 3, 4, 5]    # plausible range for bed-confined (days)

# We will select median/literature-central plausible values, but also report if the ranges for sensitivity

# --- 2. Set up rates as inverses of durations (CTMC style, so rates are per-day)
import itertools
import numpy as np

# Generate grid of plausible parameter sets
parameter_sets = []
for latent, infectious, bed in itertools.product(latent_period_choices, infectious_period_choices, bed_period_choices):
    sigma = 1.0 / latent
    kappa = 1.0 / infectious
    gamma = 1.0 / bed
    beta = R0 * kappa   # recall R0 = beta / kappa --> beta = R0 * kappa
    # Save
    param = {
        "beta": round(beta, 4),  # to 4 decimals
        "sigma": round(sigma, 4),
        "kappa": round(kappa, 4),
        "gamma": round(gamma, 4),
        "latent_period": latent, 
        "infectious_period": infectious, 
        "bed_period": bed
    }
    parameter_sets.append(param)

# For main output: select canonical (median, literature) values:
# Often: latent=1.5 days, infectious=1.5 days, bed=3 days (use closest from above)
def closest_choice(val, choices):
    array = np.array(choices)
    idx = (np.abs(array - val)).argmin()
    return choices[idx]
main_latent = closest_choice(1.5, latent_period_choices)
main_infectious = closest_choice(1.5, infectious_period_choices)
main_bed = closest_choice(3, bed_period_choices)

main_sigma = 1.0 / main_latent
main_kappa = 1.0 / main_infectious
main_gamma = 1.0 / main_bed
main_beta = R0 * main_kappa

main_params = {
    "beta": round(main_beta, 4),
    "sigma": round(main_sigma, 4),
    "kappa": round(main_kappa, 4),
    "gamma": round(main_gamma, 4)
}
# Main initial conditions:
S0 = N - 1
E0 = 0
I0 = 1
B0 = 0
C0 = 0
initial_cond = {
    "S": round(100.0 * S0 / N),   # percentage, integer
    "E": 0,
    "I": round(100.0 * I0 / N),
    "B": 0,
    "C": 0
}
# Adjustment (ensure sum to 100):
perc_sum = initial_cond["S"] + initial_cond["E"] + initial_cond["I"] + initial_cond["B"] + initial_cond["C"]
if perc_sum < 100:
    initial_cond["S"] += 100 - perc_sum
elif perc_sum > 100:
    # For the (rare) case round-up exceeded, prune susceptible
    initial_cond["S"] -= perc_sum - 100

output = {
    "main_params": main_params,
    "parameter_sets": parameter_sets,
    "initial_condition": initial_cond
}
output