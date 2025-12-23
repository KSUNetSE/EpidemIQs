
import numpy as np
from scipy import stats

# Convert systemic to int
systemic_int_2 = df2['systemic'].astype(int)

# Probability of systemic event
prob_systemic_event_percent_2 = systemic_int_2.mean() * 100

# Final cascade size statistics
mean_cascade_size_2 = df2['n_final_failed'].mean()
median_cascade_size_2 = df2['n_final_failed'].median()
std_cascade_size_2 = df2['n_final_failed'].std()
mode_res_2 = stats.mode(df2['n_final_failed'])
mode_cascade_size_2 = mode_res_2.mode.item()

# Rounds statistics
mean_rounds_2 = df2['rounds'].mean()
median_rounds_2 = df2['rounds'].median()

(prob_systemic_event_percent_2, mean_cascade_size_2, median_cascade_size_2, std_cascade_size_2, mode_cascade_size_2, mean_rounds_2, median_rounds_2)