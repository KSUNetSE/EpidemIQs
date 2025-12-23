
# Normalize states by N
cols = ['S', 'E', 'I', 'R']
for col in cols:
    data[col] = data[col] / N

# 1. Epidemic duration: time until I < 1/N = 0.0005
threshold = 1/N
times_below_threshold = data.loc[data['I'] < threshold, 'time']
epidemic_duration = times_below_threshold.iloc[0] if not times_below_threshold.empty else np.nan

# 2. Peak infection prevalence and time to peak
peak_index = data['I'].idxmax()
peak_infection = data.loc[peak_index, 'I']
time_to_peak = data.loc[peak_index, 'time']

# 3. Final epidemic size (R at last time step)
final_epidemic_size = data.loc[data.index[-1], 'R']

# 4. Initial doubling time
# Use first ~7 points or until I > 10 individuals (0.005 normalized)
early_data = data[data['I'] <= 10/N].head(7)

def exp_growth(t, r, I0):
    return I0 * np.exp(r*t)

try:
    popt, _ = curve_fit(exp_growth, early_data['time'], early_data['I'], p0=(1, early_data['I'].iloc[0]))
    r = popt[0]
    doubling_time = np.log(2) / r if r > 0 else np.nan
except:
    doubling_time = np.nan

# Compile results
results = {
    'epidemic_duration_days': epidemic_duration,
    'peak_infection_prevalence_fraction': peak_infection,
    'time_to_peak_days': time_to_peak,
    'final_epidemic_size_fraction': final_epidemic_size,
    'initial_doubling_time_days': doubling_time
}

results