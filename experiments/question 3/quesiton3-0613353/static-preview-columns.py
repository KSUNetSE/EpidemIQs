
# Print/preview the column names to debug csv extraction
import pandas as pd
import os

csv_path = os.path.join(os.getcwd(), 'output', 'results-21.csv')
data = pd.read_csv(csv_path)
columns = list(data.columns)
