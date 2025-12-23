
import numpy as np

N = 10000

# Calculate proportions
prop_R = data['R'] / N
prop_I = data['I'] / N
prop_S = data['S'] / N

time = data['time']

# 1. Final size (max R/N)
final_size = prop_R.iloc[-1]

# 2. Peak infection prevalence (max I/N)
peak_infection_prevalence = prop_I.max()
peak_infection_index = prop_I.idxmax()

# 3. Time of peak (time at max I)
time_of_peak = time.iloc[peak_infection_index]

# 4. Epidemic duration - define as time until I drops below a small threshold (eg 0.001)
threshold = 0.001
infected_above_threshold = prop_I[prop_I > threshold]
epidemic_duration = infected_above_threshold.index[-1] if not infected_above_threshold.empty else np.nan
if not np.isnan(epidemic_duration):
    epidemic_duration = time.iloc[infected_above_threshold.index[-1]] - time.iloc[0]

# 5. Doubling time in early phase
# Early phase assumed as where I is increasing and relatively low
# Take first 2 indices after start, use log2 growth to estimate doubling time if possible
early_phase_mask = (time < time_of_peak) & (prop_I > 0)
early_phase_time = time[early_phase_mask]
early_phase_I = prop_I[early_phase_mask]

# Only keep points until max I in early phase
idx_early_end = early_phase_I.idxmax() if not early_phase_I.empty else None
if idx_early_end is not None:
    early_phase_time = early_phase_time[early_phase_time.index <= idx_early_end]
    early_phase_I = early_phase_I[early_phase_I.index <= idx_early_end]


# Doubling time calculation using linear regression on log2(I) vs time, only if sufficient data
if len(early_phase_I) > 1:
    from scipy.stats import linregress
    log2_I = np.log2(early_phase_I)
    slope, intercept, r_value, p_value, std_err = linregress(early_phase_time, log2_I)
    doubling_time = 1 / slope if slope > 0 else np.nan
else:
    doubling_time = np.nan

final_size, peak_infection_prevalence, time_of_peak, epidemic_duration, doubling_time
import numpy as np

N = 10000

# Calculate final size (max R / N)
final_size = data['R'].max() / N

# Calculate peak infection prevalence (max I / N)
peak_infection_prevalence = data['I'].max() / N

# Find time of peak infection (time at which I is max)
time_of_peak = data.loc[data['I'].idxmax(), 'time']

# Define infection threshold for epidemic duration
infection_threshold = 1  # At least 1 infected individual

# Find start time of epidemic (first time I > threshold)
start_time = data.loc[data['I'] > infection_threshold, 'time'].min()

# Find end time of epidemic (last time I > threshold)
end_time = data.loc[data['I'] > infection_threshold, 'time'].max()

epidemic_duration = end_time - start_time

# Estimate doubling time during early phase
# Consider early phase as first 5% of data after start_time
early_phase_data = data[(data['time'] >= start_time) & (data['time'] <= start_time + (end_time-start_time)*0.05)]

# Only consider I values > 0
early_I = early_phase_data['I'][early_phase_data['I'] > 0]
early_time = early_phase_data['time'][early_phase_data['I'] > 0]

# Use linear regression on log(I) to estimate growth rate
from scipy.stats import linregress

slope, intercept, r_value, p_value, std_err = linregress(early_time, np.log(early_I))
# Doubling time = ln(2)/slope
if slope > 0:
    doubling_time = np.log(2) / slope
else:
    doubling_time = np.nan  # Can't calculate doubling time if no positive slope

final_size, peak_infection_prevalence, time_of_peak, epidemic_duration, doubling_time