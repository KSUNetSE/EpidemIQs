
import numpy as np

N = 1000

# Mean infected peak and time of peak
I_peak = data['I'].max()
t_peak = data.loc[data['I'].idxmax(), 'time']

# Final susceptible count (S_inf) as the last value of the series
S_inf = data['S'].iloc[-1]

# Epidemic size is N - S_inf
epidemic_size = N - S_inf

# Duration: time when I falls below a small threshold (e.g., 1) after the peak
post_peak_data = data[data['time'] > t_peak]

# Find last time I > 1
infected_threshold = 1
infected_times = post_peak_data[post_peak_data['I'] > infected_threshold]['time']
if not infected_times.empty:
    duration = infected_times.iloc[-1]
else:
    duration = t_peak  # If never falls below threshold, duration is peak time

# Extract 90% CI values for key points
S_90ci_final = (data['S_90ci_lower'].iloc[-1], data['S_90ci_upper'].iloc[-1])
I_peak_90ci = (data['I_90ci_lower'].max(), data['I_90ci_upper'].max())
I_t_peak_90ci = (data.loc[data['I'].idxmax(),'I_90ci_lower'], data.loc[data['I'].idxmax(),'I_90ci_upper'])

# As this is mean data over runs, spread in epidemic outcomes can be partially inferred from CI bands
# Residual S variation from CI bounds at final time
residual_S_spread = S_90ci_final[1] - S_90ci_final[0]

I_peak, t_peak, S_inf, epidemic_size, duration, S_90ci_final, I_peak_90ci, residual_S_spread