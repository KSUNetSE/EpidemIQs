
import pandas as pd
peak_info_all = {}
# Extract peak and duration metrics for B(t) from each results CSV (results-11.csv through results-16.csv)
for i in range(11,17):
    file = f'output/results-{i}.csv'
    df = pd.read_csv(file)
    peak_B = df['B'].max()
    peak_time_B = df.loc[df['B'].idxmax(), 'time']
    b_nonzero_times = df.loc[df['B'] > 0, 'time']
    duration_B = b_nonzero_times.max() - b_nonzero_times.min() if len(b_nonzero_times) > 1 else 0
    # CI width at peak for B
    peak_idx = df['B'].idxmax()
    B_90ci_width_at_peak = df.loc[peak_idx, 'B_90ci_upper'] - df.loc[peak_idx, 'B_90ci_lower']
    peak_info_all[file] = {
        'peak_B': peak_B,
        'peak_time_B': peak_time_B,
        'duration_B': duration_B,
        'B_90ci_width_at_peak': B_90ci_width_at_peak
    }
peak_info_all