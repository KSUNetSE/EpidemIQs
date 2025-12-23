
import numpy as np

# Constants
N = 763

# Use the results_11 dataset for initial extraction
df_11 = results_11.copy()

# Convert continuous time to integer days for daily aggregation
# time is in days, we round to nearest day
df_11['day'] = df_11['time'].round().astype(int)

# Aggregate by day: mean values for B, R, and confidence intervals
daily_11 = df_11.groupby('day').agg({
    'B': 'mean', 'B_lower': 'mean', 'B_upper': 'mean',
    'R': 'mean', 'R_lower': 'mean', 'R_upper': 'mean'}).reset_index()

# Final attack rate (AR = final R / N)
final_R = daily_11['R'].iloc[-1]
AR_11 = final_R / N

# Peak value of B and timing
peak_idx = daily_11['B'].idxmax()
peak_B = daily_11.loc[peak_idx, 'B']
peak_day = daily_11.loc[peak_idx, 'day']

# Epidemic duration: from first day with B > 0 till last day B > small threshold (e.g. 0.1)
days_with_B = daily_11[daily_11['B'] > 0.1]['day']
epidemic_duration = days_with_B.iloc[-1] - days_with_B.iloc[0] if len(days_with_B) > 1 else 0

# Display initial extracted metrics
metrics_11 = {
    'final_attack_rate': AR_11,
    'peak_B': peak_B,
    'peak_day': peak_day,
    'epidemic_duration': epidemic_duration,
    'daily_timeseries': daily_11
}

metrics_11