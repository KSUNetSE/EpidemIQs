
import numpy as np

N = 1000  # total population

# Mean peak infected and its time
peak_idx = np.argmax(data['I'])
I_peak_mean = data.loc[peak_idx, 'I']
t_peak_mean = data.loc[peak_idx, 'time']

# 90% CI for infected over time (to estimate variability at peak approximately)
I_peak_lower = data['I_90ci_lower'].max()
I_peak_upper = data['I_90ci_upper'].max()

# Final state mean and 90% CI for susceptibles
final_row = data.iloc[-1]
S_inf_mean = final_row['S']
S_inf_lower = final_row['S_90ci_lower']
S_inf_upper = final_row['S_90ci_upper']

# Epidemic size mean and 90% CI
epidemic_size_mean = N - S_inf_mean
epidemic_size_lower = N - S_inf_upper  # lower bound of epidemic size
epidemic_size_upper = N - S_inf_lower  # upper bound

# Epidemic duration = latest time when mean infected above threshold (e.g. 1 individual)
I_threshold = 1
infected_above_thresh = data[data['I'] > I_threshold]
epidemic_duration_mean_days = infected_above_thresh['time'].max() if not infected_above_thresh.empty else 0

metrics = {
    'I_peak_mean': I_peak_mean,
    'I_peak_90ci': (I_peak_lower, I_peak_upper),
    't_peak_mean': t_peak_mean,
    'S_inf_mean': S_inf_mean,
    'S_inf_90ci': (S_inf_lower, S_inf_upper),
    'epidemic_size_mean': epidemic_size_mean,
    'epidemic_size_90ci': (epidemic_size_lower, epidemic_size_upper),
    'epidemic_duration_mean_days': epidemic_duration_mean_days,
}

metrics