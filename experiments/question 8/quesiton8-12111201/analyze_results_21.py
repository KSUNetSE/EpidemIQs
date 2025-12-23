
import pandas as pd

file_path = 'output/results-21.csv'
df21 = pd.read_csv(file_path)

results_21 = analyze_posting(df21)