
# Fix indentation and re-run metric extraction in smaller steps
import numpy as np
from scipy.optimize import curve_fit

# Helper functions split and indentation fixed

def find_peak(data):
    max_I = data['I'].max()
    time_at_max_I = data.loc[data['I'].idxmax(), 'time']
    return max_I, time_at_max_I

def final_epidemic_size(data, N):
    final_R = data['R'].iloc[-1]
    fraction_R = final_R / N
    return final_R, fraction_R

def epidemic_duration(data):
    nonzero_I = data[data['I'] > 0]
    if nonzero_I.empty:
        return 0
    duration = nonzero_I['time'].iloc[-1] - nonzero_I['time'].iloc[0]
    return duration

def time_to_half_final_size(data, final_R):
    half_R = final_R / 2
    above_half = data[data['R'] >= half_R]
    if above_half.empty:
        return np.nan
    time_half = above_half['time'].iloc[0]
    return time_half

def fit_exponential_growth(data):
    early_data = data[(data['I'] > 0) & (data['R'] < 0.1 * data['R'].max())].copy()
    early_data = early_data[early_data['I'] > 0]
    if len(early_data) < 3:
        return np.nan
    
    def exp_model(t, r, I0):
        return I0 * np.exp(r * t)

    t = early_data['time'] - early_data['time'].iloc[0]
    I = early_data['I']

    try:
        params, _ = curve_fit(exp_model, t, I, p0=[0.1, I.iloc[0]], maxfev=10000)
        r = params[0]
    except Exception:
        r = np.nan
    return r

# Extract metrics
N = 1000
metrics_11 = {}
metrics_11['final_epidemic_size_number'], metrics_11['final_epidemic_size_fraction'] = final_epidemic_size(data_11, N)
metrics_11['peak_prevalence'], metrics_11['time_peak_prevalence'] = find_peak(data_11)
metrics_11['epidemic_duration'] = epidemic_duration(data_11)
metrics_11['time_to_half_final_size'] = time_to_half_final_size(data_11, metrics_11['final_epidemic_size_number'])
metrics_11['initial_growth_rate'] = fit_exponential_growth(data_11)

metrics_12 = {}
metrics_12['final_epidemic_size_number'], metrics_12['final_epidemic_size_fraction'] = final_epidemic_size(data_12, N)
metrics_12['peak_prevalence'], metrics_12['time_peak_prevalence'] = find_peak(data_12)
metrics_12['epidemic_duration'] = epidemic_duration(data_12)
metrics_12['time_to_half_final_size'] = time_to_half_final_size(data_12, metrics_12['final_epidemic_size_number'])
metrics_12['initial_growth_rate'] = fit_exponential_growth(data_12)