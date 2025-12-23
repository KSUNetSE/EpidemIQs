
import pandas as pd

# Load the data from the CSV files
results_11 = pd.read_csv('output/results-11.csv')
results_12 = pd.read_csv('output/results-12.csv')
results_13 = pd.read_csv('output/results-13.csv')

# Display the first few rows and columns info to understand the structure
results_11_head = results_11.head()
results_11_info = results_11.info()
results_12_head = results_12.head()
results_12_info = results_12.info()
results_13_head = results_13.head()
results_13_info = results_13.info()

results_11_head, results_11_info, results_12_head, results_12_info, results_13_head, results_13_info