
# Load full data from one of the files to check number of rows and uniqueness of data
full_df_21 = pd.read_csv('output/results-21.csv')

# Check the number of rows and inspect tail to understand the full time range
full_df_21_info = {
    'shape': full_df_21.shape,
    'tail': full_df_21.tail(),
    'head': full_df_21.head()
}

full_df_21_info