
import numpy as np

# Population size (since fractions are given, population N=1 for these metrics)
N = 1

# Extract final epidemic size: mean and SD of R at final time
final_R_mean = data['R_mean'].iloc[-1]
final_R_std = data['R_std'].iloc[-1]

# Peak infection rate: max I_mean and time it occurs
peak_I = data['I_mean'].max()
time_peak_I = data['t'][data['I_mean'].idxmax()]

# Time to peak: same as time of peak I
time_to_peak = time_peak_I

# Epidemic duration: first time (step) after initialization where I_mean hits 0
# Using a threshold close to 0 due to floating point
threshold = 1e-6
epidemic_duration_indices = data.index[data['I_mean'] <= threshold].tolist()
# Ensure first 0 after start (not initial 0)
epidemic_duration = None
for idx in epidemic_duration_indices:
    if data['t'][idx] > 0:
        epidemic_duration = data['t'][idx]
        break

# Without individual runs, estimate probability of major outbreak as follows:
# If final_R_mean > 1%, then there was a major outbreak in mean sense.
prob_major_outbreak = 1.0 if final_R_mean > 0.01 else 0.0

# Mean and SD epidemic duration for major outbreaks
# Can't calculate without individual runs, so provide NaN or estimate
mean_duration_major = epidemic_duration if prob_major_outbreak==1.0 else np.nan
std_duration_major = np.nan

metrics = {
    'final_R_mean': final_R_mean,
    'final_R_std': final_R_std,
    'peak_I': peak_I,
    'time_peak_I': time_peak_I,
    'time_to_peak': time_to_peak,
    'epidemic_duration': epidemic_duration,
    'prob_major_outbreak': prob_major_outbreak,
    'mean_duration_major': mean_duration_major,
    'std_duration_major': std_duration_major
}
metrics