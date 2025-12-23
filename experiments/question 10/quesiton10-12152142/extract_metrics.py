
peak_prevalence = data['peak_prevalence'].iloc[0]
time_to_peak = data['time_to_peak'].iloc[0]
time_to_10pct = data['time_to_10pct'].iloc[0]
time_to_20pct = data['time_to_20pct'].iloc[0]

# Check for additional data in the file, such as final epidemic size or duration, by describing or inspecting further
summary_stats = data.describe(include='all')