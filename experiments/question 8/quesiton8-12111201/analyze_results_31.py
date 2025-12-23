
import pandas as pd

file_path = 'output/results-31.csv'
df31 = pd.read_csv(file_path)

results_31 = analyze_posting(df31)