
# Check unique values and samples for 'final_failed' and 'is_global_cascade' in 'output/results-11.csv'
df_sample = dataframes['output/results-11.csv']
unique_final_failed = df_sample['final_failed'].unique()
unique_is_global_cascade = df_sample['is_global_cascade'].unique()
sample_data = df_sample.head()

unique_final_failed, unique_is_global_cascade, sample_data