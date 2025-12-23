
# Check if there are multiple scenarios or grouping columns
unique_columns = [col for col in data.columns if data[col].nunique() < 20 and data[col].dtype != 'float64']

# Filter data for time points around key periods: 0-10 days for early growth, near 200 days for final size
sampled_data_early = data[(data['time'] >= 0) & (data['time'] <= 10)]
sampled_data_final = data[(data['time'] >= 190) & (data['time'] <= 210)]

# Check some stats for these periods
info_early = sampled_data_early[['time', 'I']].describe()
info_final = sampled_data_final[['time', 'R']].describe()

unique_columns, info_early, info_final