
# Now, let's extract the required metrics for each run for both scenarios and check if any systemic failure (>50% failed) happened.

summary_10_details = summary_10['details']
summary_11_details = summary_11['details']

# Add a column to indicate if the run resulted in systemic failure (>50% failed)
summary_10_details['systemic_failure_50pct'] = summary_10_details['frac_failed'] > 0.5
summary_11_details['systemic_failure_50pct'] = summary_11_details['frac_failed'] > 0.5

# Extract required metrics for scenario 0 and scenario 1
scenario_0_metrics = summary_10_details[['run', 'frac_failed', 'systemic_failure_50pct', 'tsteps']]
scenario_1_metrics = summary_11_details[['run', 'frac_failed', 'systemic_failure_50pct', 'tsteps']]