
# Check if there is a run identifier column or if data is concatenated across runs
# We suspect the data might be concatenated as time series for multiple runs
# Check for unique time values and if time resets
unique_times = data['time'].unique()
# If runs are concatenated, time might reset to zero for each run
# Find indexes where time resets to zero (start of new run)
start_indexes = data.index[data['time'] == 0].tolist()
num_runs = len(start_indexes)

# Extract max infected per run and check their values
max_I_per_run = []
for i in range(num_runs):
    start_idx = start_indexes[i]
    if i == num_runs - 1:
        end_idx = len(data)
    else:
        end_idx = start_indexes[i+1]
    run_data = data.iloc[start_idx:end_idx]
    max_I_per_run.append(run_data['I'].max())

max_I_per_run = pd.Series(max_I_per_run)

# Define threshold to decide early fade-out, e.g., max I less than 5 (0.5% of population)
threshold = 5
fraction_early_fadeout = (max_I_per_run < threshold).mean()