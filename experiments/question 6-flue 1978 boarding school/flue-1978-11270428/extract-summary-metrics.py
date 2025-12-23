
# Extract relevant metrics from each summary file
results = {}

for path in file_paths:
    df = pd.read_csv(path)
    row = df.iloc[0]
    final_attack_rate_percent = row['final_attack_rate_model'] * 100
    peak_Q_size = row['peak_Q_size']
    peak_Q_time = row['peak_Q_time']
    model_R0 = row['model_R0']
    empirical_R0 = row['empirical_R0']
    N = row['N']

    # Epidemic duration is not available in summary, mark as None or Not reported
    epidemic_duration = None

    results[path] = {
        'final_attack_rate_percent': final_attack_rate_percent,
        'peak_Q_size': peak_Q_size,
        'peak_Q_time': peak_Q_time,
        'model_R0': model_R0,
        'empirical_R0': empirical_R0,
        'population': N,
        'epidemic_duration': epidemic_duration
    }

results