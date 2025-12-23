
# We have parameters and metrics for results-11 scenario
# Extract MSE between simulated B(t) and observed B(t) from csv metrics file
mse_b_11 = data_dict['output/results-11-metrics.csv'].set_index('Unnamed: 0').loc['mse_B','0']
final_ar_11 = data_dict['output/results-11-metrics.csv'].set_index('Unnamed: 0').loc['final_attack_rate','0']

# Check AR from final S and N using formula AR = 1 - S_final/N to verify consistency
S_final_11 = data_dict['output/results-11-metrics.csv'].set_index('Unnamed: 0').loc['S_final','0']
N_11 = data_dict['output/results-11-metrics.csv'].set_index('Unnamed: 0').loc['N','0']
computed_ar_11 = 1 - S_final_11/N_11

# Return values
{'mse_B_11': mse_b_11, 'final_ar_11': final_ar_11, 'computed_ar_11': computed_ar_11}