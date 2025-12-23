
import pandas as pd

# File paths
files = ['/Users/hosseinsamaei/phd/epidemiqs/output/results-00.csv', '/Users/hosseinsamaei/phd/epidemiqs/output/results-03.csv', '/Users/hosseinsamaei/phd/epidemiqs/output/results-04.csv', '/Users/hosseinsamaei/phd/epidemiqs/output/results-05.csv', '/Users/hosseinsamaei/phd/epidemiqs/output/results-06.csv', '/Users/hosseinsamaei/phd/epidemiqs/output/results-07.csv', '/Users/hosseinsamaei/phd/epidemiqs/output/results-08.csv', '/Users/hosseinsamaei/phd/epidemiqs/output/results-09.csv', '/Users/hosseinsamaei/phd/epidemiqs/output/results-10.csv', '/Users/hosseinsamaei/phd/epidemiqs/output/results-11.csv', '/Users/hosseinsamaei/phd/epidemiqs/output/results-12.csv']

# Read headers and first few rows from each file to understand structure
data_samples = {}
for f in files:
    df = pd.read_csv(f)
    data_samples[f] = {'columns': df.columns.tolist(), 'head': df.head(3)}
data_samples