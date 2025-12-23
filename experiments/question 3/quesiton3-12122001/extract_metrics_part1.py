
import numpy as np

data['I'] = data['I'].astype(float)
data['R'] = data['R'].astype(float)

# Population size
N = 1000

# 1) Epidemic Duration: time from 0 until I ~ 0 (using a threshold to be robust to noise, say I < 1)
threshold_I = 1
infected_nonzero = data[data['I'] >= threshold_I]
epidemic_duration = infected_nonzero['time'].max() if not infected_nonzero.empty else 0

# 2) Peak infection max I and percentage
peak_index = data['I'].idxmax()
peak_I = data.loc[peak_index, 'I']
peak_I_90ci_lower = data.loc[peak_index, 'I_90ci_lower']
peak_I_90ci_upper = data.loc[peak_index, 'I_90ci_upper']
peak_I_pct = peak_I / N * 100

# 3) Time to peak
peak_time = data.loc[peak_index, 'time']

# 4) Final epidemic size (R_final and attack rate)
R_final = data['R'].iloc[-1]
R_final_90ci_lower = data['R_90ci_lower'].iloc[-1]
R_final_90ci_upper = data['R_90ci_upper'].iloc[-1]
attack_rate = R_final / N * 100

epidemic_duration, peak_I, peak_I_pct, peak_I_90ci_lower, peak_I_90ci_upper, peak_time, R_final, attack_rate, R_final_90ci_lower, R_final_90ci_upper