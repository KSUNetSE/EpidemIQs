
# Check number of unique time points and a sample
unique_times = data['time'].unique()
num_unique_times = len(unique_times)
sample_unique_times = unique_times[:10]

# Check for run/replicate identifier columns
run_id_cols = [col for col in data.columns if 'run' in col.lower() or 'replicate' in col.lower()]

num_unique_times, sample_unique_times, run_id_cols