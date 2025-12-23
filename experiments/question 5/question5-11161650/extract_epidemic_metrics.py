
import numpy as np

# Function to extract epidemic metrics from one dataframe
def extract_metrics(df):
    # Initial counts at time=0
    initial_S = df.loc[df['time'] == 0, 'S'].values[0]
    initial_I = df.loc[df['time'] == 0, 'I'].values[0]
    initial_R = df.loc[df['time'] == 0, 'R'].values[0]
    N = initial_S + initial_I + initial_R

    # Final recovered at last time point
    final_R = df['R'].iloc[-1]

    # Final epidemic size (fraction)
    final_epidemic_size = (final_R - initial_R) / N

    # Epidemic peak (max number of I) and time of peak
    max_I = df['I'].max()
    time_of_peak = df.loc[df['I'] == max_I, 'time'].values[0]
    epidemic_peak_fraction = max_I / N

    # Epidemic duration: time from first I>0 to last I=0 after peak
    # Find first time I>0
    first_nonzero_I_time = df.loc[df['I'] > 0, 'time'].iloc[0]
    # Find last time I>0
    last_nonzero_I_time = df.loc[df['I'] > 0, 'time'].iloc[-1]
    epidemic_duration = last_nonzero_I_time - first_nonzero_I_time

    # Initial vaccinated (initial R)
    initial_vaccinated = initial_R

    return {
        'initial_S': initial_S,
        'initial_I': initial_I,
        'initial_R': initial_R,
        'N': N,
        'final_epidemic_size': final_epidemic_size,
        'epidemic_peak_fraction': epidemic_peak_fraction,
        'time_of_peak': time_of_peak,
        'epidemic_duration': epidemic_duration,
        'initial_vaccinated': initial_vaccinated
    }

# Extract metrics for all files
data_metrics = []
for file_path in file_paths:
    df = pd.read_csv(file_path)
    metrics = extract_metrics(df)
    metrics['file'] = file_path
    data_metrics.append(metrics)

# Convert to DataFrame for better presentation
metrics_df = pd.DataFrame(data_metrics)