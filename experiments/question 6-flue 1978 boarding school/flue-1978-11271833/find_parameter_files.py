
# We have peak prevalence, peak time, duration and CI width for B for results-11 as an example
# Now checking if parameter values (beta, sigma, gamma, delta, kappa, seed) are in any accessible files
# Searching for parameter files (looking also for results-11-parameters.csv or similar)
param_files = [f for f in files_in_output if 'param' in f or 'parameter' in f]
param_files