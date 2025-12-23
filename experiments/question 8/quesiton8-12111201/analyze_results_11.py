
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Define a function to analyze a single dataframe for the requested metrics

def analyze_posting(df):
    time = df['time'].values
    P = df['P'].values
    
    # 1. Peak Posting Rate (max P and its time)
    max_peak_idx = np.argmax(P)
    peak_posting_rate = P[max_peak_idx]
    peak_posting_time = time[max_peak_idx]

    # 2. Final endemic level of Posting (mean P over last 100 days)
    last_100_days_mask = time >= (time[-1] - 100)
    final_endemic_posting = np.mean(P[last_100_days_mask])

    # 3. Amplitude of Posting in last 100 days (max-min)
    amplitude_last_100 = np.max(P[last_100_days_mask]) - np.min(P[last_100_days_mask])
    amplitude_last_100_std = np.std(P[last_100_days_mask])

    # 4. Damping rate (ratio of first to second peak amplitude)
    peaks, _ = find_peaks(P)
    if len(peaks) >= 2:
        first_peak_amplitude = P[peaks[0]]
        second_peak_amplitude = P[peaks[1]]
        damping_rate = first_peak_amplitude / second_peak_amplitude if second_peak_amplitude != 0 else np.nan
    else:
        damping_rate = np.nan  # Not enough peaks to calculate damping

    # 5. Epidemic duration (time from initial rise to when P remains within 5% of endemic level)
    initial_rise_idx = np.where(P > P[0])[0][0] if np.any(P > P[0]) else 0
    endemic_level_band_lower = final_endemic_posting * 0.95
    endemic_level_band_upper = final_endemic_posting * 1.05
    # Find first time when P remains within 5% band continuously until end
    epidemic_end_idx = None
    for i in range(initial_rise_idx, len(P)):
        if np.all((P[i:] >= endemic_level_band_lower) & (P[i:] <= endemic_level_band_upper)):
            epidemic_end_idx = i
            break
    epidemic_duration = time[epidemic_end_idx] - time[initial_rise_idx] if epidemic_end_idx is not None else np.nan

    # 6. Cumulative Posting (time integral/area under curve)
    cumulative_posting = np.trapz(P, time)

    # 7. Population dynamics time series (return dataframe columns U, P, F and time for validation)
    population_dynamics = df[['time', 'U', 'P', 'F']]

    results = {
        'peak_posting_rate': peak_posting_rate,
        'peak_posting_time': peak_posting_time,
        'final_endemic_posting': final_endemic_posting,
        'amplitude_last_100': amplitude_last_100,
        'amplitude_last_100_std': amplitude_last_100_std,
        'damping_rate': damping_rate,
        'epidemic_duration': epidemic_duration,
        'cumulative_posting': cumulative_posting,
        'population_dynamics': population_dynamics
    }

    return results

# Test the function on one loaded file df from earlier
results_11 = analyze_posting(df)