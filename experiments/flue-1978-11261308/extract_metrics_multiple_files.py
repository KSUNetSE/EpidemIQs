
# Repeat extraction for results_12, results_13, results_16 datasets

metrics = {}

for key, df in [('results_12', results_12), ('results_13', results_13), ('results_16', results_16)]:
    df['day'] = df['time'].round().astype(int)
    daily = df.groupby('day').agg({
        'B': 'mean', 'B_lower': 'mean', 'B_upper': 'mean',
        'R': 'mean', 'R_lower': 'mean', 'R_upper': 'mean'}).reset_index()
    final_R = daily['R'].iloc[-1]
    AR = final_R / N
    peak_idx = daily['B'].idxmax()
    peak_B = daily.loc[peak_idx, 'B']
    peak_day = daily.loc[peak_idx, 'day']
    days_with_B = daily[daily['B'] > 0.1]['day']
    epidemic_duration = days_with_B.iloc[-1] - days_with_B.iloc[0] if len(days_with_B) > 1 else 0
    metrics[key] = {
        'final_attack_rate': AR,
        'peak_B': peak_B,
        'peak_day': peak_day,
        'epidemic_duration': epidemic_duration,
        'daily_timeseries': daily
    }

metrics