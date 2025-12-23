
# Extract parameter dictionary by executing script portion
exec(param_script)

# Return extracted params and initial conditions
param_summary = {
    'beta': beta,
    'sigma': sigma,
    'gamma': gamma,
    'delta': delta,
    'kappa': kappa,
    'R0_from_params': R0,
    'mean_generation_time_days': gen_time,
    'population_n': n,
    'seed_info': '3 initial infectious nodes (all in I) in one house'
}
param_summary