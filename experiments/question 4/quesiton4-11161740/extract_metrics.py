
import numpy as np

def extract_metrics(df):
    # Last 20% of time to calculate steady state
    last_20pct_index = int(len(df)*0.8)
    tail_df = df.iloc[last_20pct_index:]
    
    # Steady-state prevalence mean and 90% CI for I1 and I2
    I1_ss_mean = tail_df['I1_mean'].mean()
    I1_ss_ci_lower = tail_df['I1_90ci_lower'].mean()
    I1_ss_ci_upper = tail_df['I1_90ci_upper'].mean()
    I2_ss_mean = tail_df['I2_mean'].mean()
    I2_ss_ci_lower = tail_df['I2_90ci_lower'].mean()
    I2_ss_ci_upper = tail_df['I2_90ci_upper'].mean()

    # Peak prevalence for I1 and I2
    I1_peak = df['I1_mean'].max()
    I2_peak = df['I2_mean'].max()

    # Time to steady state: relative change <1e-3 over last 5% window (search starting from earliest time)
    rel_change_thresh = 1e-3
    window_size = int(len(df)*0.05)

    def find_stabilization_time(series):
        for i in range(window_size, len(series)):
            # Calculate relative change over last window_size
            window = series.iloc[i-window_size:i]
            rel_changes = np.abs(window.pct_change().dropna())
            if (rel_changes < rel_change_thresh).all():
                return df['time'].iloc[i]
        return np.nan  # Not stabilized

    I1_stab_time = find_stabilization_time(df['I1_mean'])
    I2_stab_time = find_stabilization_time(df['I2_mean'])

    # Extinctions: incidence <1% prevalence threshold
    extinction_threshold = 0.01
    I1_extinct = (df['I1_mean'] < extinction_threshold).all()
    I2_extinct = (df['I2_mean'] < extinction_threshold).all()

    # Classify regime
    # Using steady state mean prevalences
    if I1_ss_mean > 0.01 and I2_ss_mean > 0.01:
        regime = 'coexistence'
    elif I1_ss_mean > 0.01 and I2_ss_mean <= 0.01:
        regime = 'dominance by 1'
    elif I2_ss_mean > 0.01 and I1_ss_mean <= 0.01:
        regime = 'dominance by 2'
    else:
        regime = 'bistable/extinct'

    metrics = {
        'I1_ss_mean': I1_ss_mean, 'I1_ss_ci_lower': I1_ss_ci_lower, 'I1_ss_ci_upper': I1_ss_ci_upper,
        'I2_ss_mean': I2_ss_mean, 'I2_ss_ci_lower': I2_ss_ci_lower, 'I2_ss_ci_upper': I2_ss_ci_upper,
        'I1_peak': I1_peak, 'I2_peak': I2_peak,
        'I1_stab_time': I1_stab_time, 'I2_stab_time': I2_stab_time,
        'I1_extinct': I1_extinct, 'I2_extinct': I2_extinct,
        'regime': regime
    }

    return metrics

all_metrics = {}

for fp in file_paths:
    df = pd.read_csv(fp)
    metrics = extract_metrics(df)
    all_metrics[fp] = metrics

all_metrics