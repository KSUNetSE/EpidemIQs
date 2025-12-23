
import pandas as pd

# Load the two datasets
results_11 = pd.read_csv('output/results-11.csv')
results_21 = pd.read_csv('output/results-21.csv')

# Check the basic structure of both datasets
results_11_shape = results_11.shape
results_21_shape = results_21.shape

# Display the first few rows to understand the columns
results_11_head = results_11.head()
results_21_head = results_21.head()