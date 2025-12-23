
import numpy as np
from scipy import stats

# Convert systemic flag to int
systemic_int_21 = df21['systemic'].astype(int)

# Calculate probability of systemic event
prob_systemic_event_21 = systemic_int_21.mean() * 100

# Final cascade size stats
mean_cascade_21 = df21['n_final_failed'].mean()
median_cascade_21 = df21['n_final_failed'].median()
std_cascade_21 = df21['n_final_failed'].std()
mode_res_21 = stats.mode(df21['n_final_failed'])
mode_cascade_21 = mode_res_21.mode.item()

# Absorption rounds stats
mean_rounds_21 = df21['rounds'].mean()
median_rounds_21 = df21['rounds'].median()

prob_systemic_event_21, mean_cascade_21, median_cascade_21, std_cascade_21, mode_cascade_21, mean_rounds_21, median_rounds_21