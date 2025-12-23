
# Check for presence of model parameters beta, gamma1, sigma in the simulation data columns or first few rows
param_search = {}
for key, df in zip(files.keys(), [results_11, results_12, results_13, results_16]):
    parameters_found = [col for col in df.columns if any(x in col.lower() for x in ['beta', 'gamma', 'sigma'])]
    param_search[key] = parameters_found
param_search
