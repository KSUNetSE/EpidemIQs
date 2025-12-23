
import pandas as pd
import numpy as np
import os

# Load simulation results
df = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'))

def get_peak_infection_metrics(df):
    # Find peak infected and its time
    peak_I = df['I'].max()
    peak_time = df.loc[df['I'].idxmax(), 'time']
    # Final epidemic size (# recovered at final time)
    final_R = df['R'].iloc[-1]
    # Doubling time: time it takes for I to go from 10->20 (if possible)
    doubling_time = np.nan
    i_start = df['I'].ge(10).idxmax()
    i2 = df['I'].ge(20).idxmax() if (df['I']>=20).any() else None
    if i2 and i2 > i_start:
        doubling_time = df['time'][i2] - df['time'][i_start]
    # Epidemic duration: time from first I>10 to when I<1
    end_idx = df['I'].le(1).to_numpy().nonzero()[0]
    if len(end_idx) > 0 and end_idx[-1] > i_start:
        duration = df['time'].iloc[end_idx[-1]] - df['time'].iloc[i_start]
    else:
        duration = np.nan
    return {
        'Peak Infection': int(peak_I),
        'Peak Time': float(peak_time),
        'Final Epidemic Size (R_inf)': int(final_R),
        'Doubling Time': float(doubling_time) if not np.isnan(doubling_time) else 'N/A',
        'Epidemic Duration': float(duration) if not np.isnan(duration) else 'N/A',
    }

metrics = get_peak_infection_metrics(df)

# Summarize for report
summary = f"""
Peak infected number: {metrics['Peak Infection']} at t={metrics['Peak Time']:.1f}\nFinal epidemic size (total recovered): {metrics['Final Epidemic Size (R_inf)']}\nDoubling time: {metrics['Doubling Time']}\nEpidemic duration: {metrics['Epidemic Duration']} time units\n"""
(summary, metrics)
import pandas as pd
import numpy as np
import os

# Load results
results = pd.read_csv(os.path.join("output","results-11.csv"))
time = results['time']
S = results['S']
I = results['I']
R = results['R']
N = S[0] + I[0] + R[0]

# Metrics
peak_I = np.max(I)
peak_I_frac = peak_I / N
peak_time = time[I.argmax()]
final_size = R.iloc[-1] / N

duration = time[(I+R > I[0]).argmax()]  # time to last infection

doubling_times = []
for t in range(len(I)//2):
    if I[t]>0 and I[t*2]>I[t]:
        doubling_times.append(time[t*2] - time[t])
doubling_time = np.mean(doubling_times) if doubling_times else np.nan

# Table
metrics = pd.DataFrame({
    'EpidemicPeak': [peak_I],
    'PeakFraction': [peak_I_frac],
    'PeakTime': [peak_time],
    'FinalSize': [final_size],
    'Duration': [duration],
    'DoublingTime': [doubling_time]
})
metrics.to_csv(os.path.join("output","metrics-11.csv"), index=False)
peak_I, peak_I_frac, peak_time, final_size, duration, doubling_time
import pandas as pd
import numpy as np

data = pd.read_csv('output/results-11.csv')

# Metrics to extract:
# 1. Epidemic duration: time until less than 1 currently infected (terminated)
# 2. Peak infection rate & size: max(I)
# 3. Final epidemic size: total R at end
# 4. Doubling time: time for I to go from first nonzero to double that value (approximation)
# 5. Peak time of infection

# 1. Epidemic duration
threshold = 1.0
if (data['I'] < threshold).any():
    t_end = data['time'][data['I'] < threshold].iloc[0]
else:
    t_end = data['time'].iloc[-1]

# 2. Peak infection
peak_I = data['I'].max()
peak_time = data['time'][data['I'].idxmax()]

# 3. Final epidemic size
final_R = data['R'].iloc[-1]

# 4. Doubling time
I_values = data['I'].values
I_start = I_values[I_values > 0][0]
if np.any(I_values > 2 * I_start):
    double_time = data['time'][np.where(I_values > 2*I_start)[0][0]] - data['time'][np.where(I_values > 0)[0][0]]
else:
    double_time = np.nan

metrics = {
    'epidemic_duration': float(t_end),
    'peak_infection': float(peak_I),
    'peak_time': float(peak_time),
    'final_epidemic_size': float(final_R),
    'doubling_time': float(double_time),
}
metrics
