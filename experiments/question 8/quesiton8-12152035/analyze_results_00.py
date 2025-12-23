
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq

# Load baseline file for detailed analysis
file = '/Users/hosseinsamaei/phd/epidemiqs/output/results-00.csv'
df = pd.read_csv(file)

# Total population size N inferred from initial data
# N should be roughly constant near max sum of U+P+F
N = (df['U'] + df['P'] + df['F']).max()

# Threshold for deciding epidemic duration (1% of N)
thresh = 0.01 * N

# 1. Peak Posting Prevalence (max P and time)
peak_P = df['P'].max()
peak_time = df.loc[df['P'].idxmax(), 'time']
peak_P_ci = (df.loc[df['P'].idxmax(), 'P_90ci_lower'], df.loc[df['P'].idxmax(), 'P_90ci_upper'])

# 2. Epidemic Duration (time with P > 1% N)
on_periods = df['P'] > thresh
if on_periods.any():
    start_time = df.loc[on_periods.idxmax(), 'time'] # first time P > thresh
    # Find last time P > thresh
    last_idx = on_periods[::-1].idxmax()
    end_time = df.loc[last_idx, 'time']
epidemic_duration = end_time - start_time

# 3. Final posting and fatigued at maximum time
tmax = df['time'].max()
final_P = df.loc[df['time'] == tmax, 'P'].values[0]
final_F = df.loc[df['time'] == tmax, 'F'].values[0]
final_P_ci = (df.loc[df['time'] == tmax, 'P_90ci_lower'].values[0], df.loc[df['time'] == tmax, 'P_90ci_upper'].values[0])
final_F_ci = (df.loc[df['time'] == tmax, 'F_90ci_lower'].values[0], df.loc[df['time'] == tmax, 'F_90ci_upper'].values[0])

# 4. Doubling time of P in early phase (from 1% N to 2% N)
P_1pct_idx = df[df['P'] >= thresh].index[0]
P_2pct_idx = df[df['P'] >= 2*thresh].index[0] if any(df['P'] >= 2*thresh) else None
if P_2pct_idx and P_2pct_idx > P_1pct_idx:
    doubling_time = df.loc[P_2pct_idx, 'time'] - df.loc[P_1pct_idx, 'time']
else:
    doubling_time = None

# 5. Detect and quantify oscillations in P (count peaks, periods, amplitude)
# Using scipy find_peaks with prominence to filter noise
peaks, properties = find_peaks(df['P'], prominence=0.01*N)  # prominence threshold adjustable
num_peaks = len(peaks)

if num_peaks > 1:
    peak_times = df.loc[peaks, 'time'].values
    periods = np.diff(peak_times)
    mean_period = np.mean(periods)
    amplitude = (np.max(df['P'].values[peaks]) - np.min(df['P'].values[peaks]))
    approach = 'oscillatory'
else:
    mean_period = None
    amplitude = None
    approach = 'steady/endemic'

# 6. Spectral analysis using FFT on P time series
# Sampling frequency estimated from median diff in time
times = df['time'].values
dt = np.median(np.diff(times)) if len(times) > 1 else 1
P_values = df['P'].values - np.mean(df['P'].values)  # demean
frequencies = rfftfreq(len(P_values), d=dt)
fft_vals = np.abs(rfft(P_values))
# Ignore zero freq
dominant_freq_idx = np.argmax(fft_vals[1:]) + 1
if dominant_freq_idx < len(frequencies):
    dominant_freq = frequencies[dominant_freq_idx]
    dominant_period = 1 / dominant_freq if dominant_freq > 0 else None
else:
    dominant_freq = None
    dominant_period = None

# Collect results in dictionary
results = {
    'file': file,
    'N': N,
    'peak_posting': peak_P,
    'peak_time': peak_time,
    'peak_P_90ci': peak_P_ci,
    'epidemic_duration_days': epidemic_duration,
    'final_posting': final_P,
    'final_fatigued': final_F,
    'final_P_90ci': final_P_ci,
    'final_F_90ci': final_F_ci,
    'doubling_time_early_days': doubling_time,
    'oscillation_num_peaks': num_peaks,
    'oscillation_mean_period_days': mean_period,
    'oscillation_amplitude': amplitude,
    'oscillation_approach': approach,
    'spectral_dominant_freq': dominant_freq,
    'spectral_dominant_period_days': dominant_period
}
results