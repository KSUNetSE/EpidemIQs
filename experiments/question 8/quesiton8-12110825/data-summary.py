
# Number of runs
num_runs = len(data)

# Count NaNs in each column
nan_counts = data.isna().sum()

# Descriptive statistics for the metrics excluding NaNs
stats = data.describe()

num_runs, nan_counts, stats