
import numpy as np
from scipy import stats

# Ensure correct types
systemic_int = df['systemic'].astype(int)

# Probability of systemic event
prob_systemic_event_percent = systemic_int.mean() * 100

# Final cascade size statistics
mean_cascade_size = df['n_final_failed'].mean()
median_cascade_size = df['n_final_failed'].median()
std_cascade_size = df['n_final_failed'].std()
mode_res = stats.mode(df['n_final_failed'])
mode_cascade_size = mode_res.mode.item()

# Rounds statistics
mean_rounds = df['rounds'].mean()
median_rounds = df['rounds'].median()

prob_systemic_event_percent, mean_cascade_size, median_cascade_size, std_cascade_size, mode_cascade_size, mean_rounds, median_rounds