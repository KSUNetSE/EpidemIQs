
# Reformat result for specification: use explicit scenario-by-scenario parameter dictionaries, key by scenario and then by case index.
# Strip taus from output (not model-rate params!), ensure keys/values only CTMC rates as floats, round for readability.
params_high = []
params_low = []

for p in results['param_table_high']:
    params_high.append({
        'beta1': round(float(p['beta1']), 5),
        'delta1': round(float(p['delta1']), 2),
        'beta2': round(float(p['beta2']), 5),
        'delta2': round(float(p['delta2']), 2)
    })
for p in results['param_table_low']:
    params_low.append({
        'beta1': round(float(p['beta1']), 5),
        'delta1': round(float(p['delta1']), 2),
        'beta2': round(float(p['beta2']), 5),
        'delta2': round(float(p['delta2']), 2)
    })

parameter_sets = {
    'high_overlap': params_high,
    'low_overlap': params_low
}

# Initial conditions are already formatted as percentage ints.
initial_conditions = results['initial_conditions']
initial_condition_desc = results['initial_condition_desc']

out_bundle = {
    'parameters': parameter_sets,
    'initial_conditions': initial_conditions,
    'initial_condition_desc': initial_condition_desc,
}
