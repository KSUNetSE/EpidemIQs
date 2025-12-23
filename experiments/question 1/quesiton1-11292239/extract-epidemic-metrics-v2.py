
import numpy as np
from scipy.stats import linregress

# 1. Epidemic duration: find time when I falls below 1 after peaking
peak_index = np.argmax(data['I'])
epidemic_end_indices = np.where(data['I'][peak_index:] < 1)[0]
if len(epidemic_end_indices) > 0:
    epidemic_duration = data['time'][peak_index + epidemic_end_indices[0]]
else:
    epidemic_duration = data['time'].iloc[-1]

# 2. Peak I and when
peak_I = data['I'][peak_index]
peak_time = data['time'][peak_index]

# 3. Final R
final_R = data['R'].iloc[-1]

# 4. Initial doubling time estimation during early exponential growth
early_mask = (data['time'] <= peak_time) & (data['I'] > 0)
early_times = data.loc[early_mask, 'time']
early_I = data.loc[early_mask, 'I']

log_early_I = np.log(early_I)
regress_result = linregress(early_times, log_early_I)
slope = regress_result.slope
if slope > 0:
    doubling_time = np.log(2) / slope
else:
    doubling_time = float('nan')

(epidemic_duration, peak_I, peak_time, final_R, doubling_time)