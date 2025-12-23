
import pandas as pd

# Load datasets
file_11 = '/Users/hosseinsamaei/phd/gemf_llm/output/results-11.csv'
file_12 = '/Users/hosseinsamaei/phd/gemf_llm/output/results-12.csv'

data_11 = pd.read_csv(file_11)
data_12 = pd.read_csv(file_12)

# Examine initial few rows and columns of both datasets
info_11 = data_11.info()
info_12 = data_12.info()
head_11 = data_11.head()
head_12 = data_12.head()

info_11, info_12, head_11, head_12