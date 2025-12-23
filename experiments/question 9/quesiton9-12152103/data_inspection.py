
import pandas as pd

# Load both CSV files to inspect their structure
file_core = '/Users/hosseinsamaei/phd/epidemiqs/output/results-10.csv'
file_periphery = '/Users/hosseinsamaei/phd/epidemiqs/output/results-11.csv'

df_core = pd.read_csv(file_core)
df_periphery = pd.read_csv(file_periphery)

# Get basic info on columns and sample data
core_info = df_core.info(buf=None)
periphery_info = df_periphery.info(buf=None)

core_head = df_core.head().to_dict()
periphery_head = df_periphery.head().to_dict()

# Extract column names and shapes
core_columns = df_core.columns.tolist()
periphery_columns = df_periphery.columns.tolist()
core_shape = df_core.shape
periphery_shape = df_periphery.shape