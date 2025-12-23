
# Check earliest infected values and infected values over time to inspect fadeout possibility
# Identify first index where time is > 0 to analyze initial infected dynamics
first_time_positive = data[data['time']>0].iloc[0]['time']

# Identify time points where infected is below threshold 10
infected_below_10 = data[data['I']<=10]['time'].tolist()

# Check confidence interval ranges for infected to see if lower bound hits zero or close
infected_90ci_low_zero_or_near = (data['I_90ci_lower'] <= 10).sum()
total_rows = data.shape[0]

first_time_positive, infected_below_10[:10], infected_90ci_low_zero_or_near, total_rows