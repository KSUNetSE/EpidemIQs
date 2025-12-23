
def get_initial_conditions(population_size, infectious, exposed):
    """
    Returns integer percentage-based initial conditions for SEIR compartments
    Input:
      - population_size (int): system size
      - infectious (int): initial infectious individuals
      - exposed (int): initial exposed individuals
    Returns:
      - Dict of {'S':%, 'E':%, 'I':%, 'R':%}, all integers summing to 100
    """
    n_inf = infectious
    n_exp = exposed
    n_sus = population_size - infectious - exposed
    n_rec = 0
    # Calculate preliminary percentages (floating)
    S_frac = n_sus / population_size
    E_frac = n_exp / population_size
    I_frac = n_inf / population_size
    R_frac = n_rec / population_size
    # Convert to percentages (no decimals)
    vals = [S_frac, E_frac, I_frac, R_frac]
    vals_pct = [int(round(v*100)) for v in vals]
    diff = 100 - sum(vals_pct)
    # Fix rounding to sum exactly 100
    # Adjust largest group (usually S) for loss/gain from rounding
    max_ix = vals_pct.index(max(vals_pct))
    vals_pct[max_ix] += diff 
    return {'S': vals_pct[0], 'E': vals_pct[1], 'I': vals_pct[2], 'R': vals_pct[3]}

# Scenario 1: random initial seeding
# N = 2000, 1 I, 3 E (as in scenario), R always 0
init_rand = get_initial_conditions(2000, 1, 3)

# Scenario 2: same percentages but description is: hub-based seeding for BA
init_hubs = dict(init_rand) # For network, will assign 1 I + 3 E to highest-k nodes (just assignment differs)

return_vars = ["init_rand", "init_hubs"]
