
import numpy as np
import matplotlib.pyplot as plt

N = 1000  # population size

# Peak mean infection and time to peak
I_peak = data['I'].max()
idx_peak = data['I'].idxmax()
t_peak = data.loc[idx_peak, 'time']
I_peak_90ci = (data.loc[idx_peak, 'I_90ci_lower'], data.loc[idx_peak, 'I_90ci_upper'])

# Final mean susceptibles is at the last time point
S_inf = data['S'].iloc[-1]
S_inf_90ci = (data['S_90ci_lower'].iloc[-1], data['S_90ci_upper'].iloc[-1])
final_epidemic_size = N - S_inf
final_epidemic_size_90ci = (N - S_inf_90ci[1], N - S_inf_90ci[0])

# Epidemic duration: mean time when mean I falls below 1 after peak
# Find all times after peak where mean I < 1
post_peak = data.loc[idx_peak:]
times_below_1 = post_peak[post_peak['I'] < 1]['time']
if len(times_below_1) > 0:
    t_end = times_below_1.iloc[0]
else:
    t_end = data['time'].iloc[-1]

# Plot mean I(t) with 90% CI
plt.fill_between(data['time'], data['I_90ci_lower'], data['I_90ci_upper'], color='lightblue', alpha=0.5, label='90% CI')
plt.plot(data['time'], data['I'], color='blue', label='Mean I')
plt.axvline(t_peak, color='red', linestyle='--', label=f'Peak time ({t_peak:.2f})')
plt.axhline(1, color='gray', linestyle=':')
plt.axvline(t_end, color='green', linestyle='--', label=f'Epidemic end ({t_end:.2f})')
plt.xlabel('Time')
plt.ylabel('Mean I')
plt.title('Mean infected over time with 90% CI')
plt.legend()
plt.tight_layout()
import os
output_dir = os.path.join(os.getcwd(), "output")
plt.savefig(os.path.join(output_dir, 'mean_I_over_time.png'))

metrics = {
    'I_peak': I_peak, 
    'I_peak_90ci': I_peak_90ci, 
    't_peak': t_peak, 
    'S_inf': S_inf, 
    'S_inf_90ci': S_inf_90ci, 
    'final_epidemic_size': final_epidemic_size, 
    'final_epidemic_size_90ci': final_epidemic_size_90ci, 
    't_end': t_end
}