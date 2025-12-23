
# Further inspection on combined data
# Sample from run 1
run1_sample = combined_data[combined_data['run'] == 1].head(3)

# Check unique runs in combined data
unique_runs = combined_data['run'].unique().tolist()

run1_sample, unique_runs