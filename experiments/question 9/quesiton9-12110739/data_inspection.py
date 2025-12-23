
import pandas as pd

# Read the first few rows and columns of each CSV to understand the structure
data_samples = {}
file_paths = ['/Users/hosseinsamaei/phd/epidemiqs/output/results-110.csv', '/Users/hosseinsamaei/phd/epidemiqs/output/results-120.csv', '/Users/hosseinsamaei/phd/epidemiqs/output/results-310.csv', '/Users/hosseinsamaei/phd/epidemiqs/output/results-320.csv', '/Users/hosseinsamaei/phd/epidemiqs/output/results-210.csv', '/Users/hosseinsamaei/phd/epidemiqs/output/results-220.csv', '/Users/hosseinsamaei/phd/epidemiqs/output/results-410.csv', '/Users/hosseinsamaei/phd/epidemiqs/output/results-420.csv']

for path in file_paths:
    df = pd.read_csv(path)
    data_samples[path] = df.head().to_dict(), df.columns.tolist(), df.shape

