
import pandas as pd

# Load the data from the two files
path_er = '/Users/hosseinsamaei/phd/epidemiqs/output/results-11.csv'
path_ba = '/Users/hosseinsamaei/phd/epidemiqs/output/results-12.csv'

df_er = pd.read_csv(path_er)
df_ba = pd.read_csv(path_ba)

# Extract the basic info about dataframes to understand their structure
info_er = df_er.info(verbose=True)
info_ba = df_ba.info(verbose=True)

# Preview first few rows to see the column names and data
head_er = df_er.head()
head_ba = df_ba.head()

info_er_str = str(info_er)
info_ba_str = str(info_ba)


# Check column names of both datasets
columns_er = df_er.columns.tolist()
columns_ba = df_ba.columns.tolist()

columns_er, columns_ba