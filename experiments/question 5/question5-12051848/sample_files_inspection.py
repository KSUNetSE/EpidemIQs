
# Specifically load a few unique pattern files to see the internal structure in detail
sample_files = ['output/results-11.csv', 'output/results-1attackrates.csv', 'output/results-4attackrates.csv']

sample_dfs = {fp: pd.read_csv(fp) for fp in sample_files}

sample_dfs_info = {fp: {
    'columns': sample_dfs[fp].columns.tolist(),
    'head': sample_dfs[fp].head(3),
    'tail': sample_dfs[fp].tail(3),
    'shape': sample_dfs[fp].shape
} for fp in sample_files}