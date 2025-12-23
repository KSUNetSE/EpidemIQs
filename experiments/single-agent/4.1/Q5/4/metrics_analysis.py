
import pandas as pd
import numpy as np
import os

def epidemic_metrics(csv_path):
    df = pd.read_csv(csv_path)
    # final size: total R at end minus initial R
    final_size = df['R'].iloc[-1] - df['R'].iloc[0]
    # peak infected
    peak_I = df['I'].max()
    time_to_peak = df['time'][df['I'].idxmax()]
    # epidemic duration until <1 infectious
    thresh = 1
    try:
        end_idx = np.where(df['I'] < thresh)[0][0]
        duration = df['time'][end_idx]
    except:
        duration = df['time'].iloc[-1]
    metrics = {
        'final_size': final_size,
        'peak_I': peak_I,
        'time_to_peak': time_to_peak,
        'duration': duration
    }
    return metrics

metrics_hom = epidemic_metrics('output/results-1-1.csv')
metrics_het = epidemic_metrics('output/results-1-2.csv')
(metrics_hom, metrics_het)
import pandas as pd
import numpy as np
import os

# Load simulation results
results_path = os.path.join(os.getcwd(),'output','results-11.csv')
data = pd.read_csv(results_path)

# Metrics to extract: Epidemic duration (last time I > threshold),
# Peak infection size, Peak time, Final epidemic size, Doubling time estimate

# 1. Epidemic Duration (when I < 1 for last time)
epidemic_ongoing = data['I'] > 1
duration_idx = np.where(epidemic_ongoing)[0]
if len(duration_idx) > 0:
    epidemic_end_time = data['time'].iloc[duration_idx[-1]]
else:
    epidemic_end_time = 0

# 2. Peak infection and its time
peak_infection = data['I'].max()
peak_time = data['time'][data['I'].idxmax()]

# 3. Final epidemic size (total ever infected: N - S(final))
N = 1000  # Must match modeling
final_susceptible = data['S'].iloc[-1]
final_epidemic_size = N - final_susceptible

# 4. Doubling time estimate: time to double from 1 to 2*initial infectives
init_I = data['I'].iloc[0]
double_I = 2 * init_I
if data['I'].max() > double_I:
    double_idx = np.where(data['I'] >= double_I)[0][0]
    doubling_time = data['time'].iloc[double_idx]
else:
    doubling_time = np.nan

metrics = {
    'epidemic_duration': epidemic_end_time,
    'peak_infection': peak_infection,
    'peak_time': peak_time,
    'final_epidemic_size': final_epidemic_size,
    'doubling_time': doubling_time
}
metrics