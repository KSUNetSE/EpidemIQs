
import pandas as pd
import os

filepaths = [os.path.join("output", f"results-{i}0.csv") for i in range(5)]

data_samples = {}
for fp in filepaths:
    data_samples[fp] = pd.read_csv(fp).head()

# Checking columns and samples from each file to understand the structure
cols_samples = {fp: (df.columns.tolist(), df.head()) for fp, df in data_samples.items()}
cols_samples