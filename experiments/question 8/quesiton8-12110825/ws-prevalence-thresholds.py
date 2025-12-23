
# Find earliest times when infected fraction exceeds thresholds 0.10 and 0.50
exceeds_10 = ws_data[ws_data['I'] > 0.10]
exceeds_50 = ws_data[ws_data['I'] > 0.50]

# Extract earliest time points if exceedances exist
first_time_10 = exceeds_10['time'].min() if not exceeds_10.empty else None
first_time_50 = exceeds_50['time'].min() if not exceeds_50.empty else None

first_time_10, first_time_50