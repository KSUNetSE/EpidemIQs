
# Load full scenario param map (already loaded as df_params from previous step but reloading for clarity)
df_params = pd.read_csv(param_file)

# Prepare mapping from CSV filename to parameters
param_map = df_params.set_index('results_csv')[['beta', 'gamma', 'xi']].to_dict(orient='index')

# Add N, k, p from knowledge/context to param_map since not provided
# N: 10000 for results-00.csv, 1000 for others
# k approx 8, p approx 0.1
for key in param_map:
    if key == 'results-00.csv':
        param_map[key]['N'] = 10000
    else:
        param_map[key]['N'] = 1000
    param_map[key]['k'] = 8
    param_map[key]['p'] = 0.1

param_map