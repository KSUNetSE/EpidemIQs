
import pandas as pd

file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-11.csv'
data_11 = pd.read_csv(file_path)

file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-12.csv'
data_12 = pd.read_csv(file_path)

file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-13.csv'
data_13 = pd.read_csv(file_path)

file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-14.csv'
data_14 = pd.read_csv(file_path)

file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-15.csv'
data_15 = pd.read_csv(file_path)

file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-16.csv'
data_16 = pd.read_csv(file_path)

# Check the columns and head for each
summary_11 = (data_11.columns.tolist(), data_11.head(), data_11.shape)
summary_12 = (data_12.columns.tolist(), data_12.head(), data_12.shape)
summary_13 = (data_13.columns.tolist(), data_13.head(), data_13.shape)
summary_14 = (data_14.columns.tolist(), data_14.head(), data_14.shape)
summary_15 = (data_15.columns.tolist(), data_15.head(), data_15.shape)
summary_16 = (data_16.columns.tolist(), data_16.head(), data_16.shape)

summary_all = {
    'results-11': summary_11,
    'results-12': summary_12,
    'results-13': summary_13,
    'results-14': summary_14,
    'results-15': summary_15,
    'results-16': summary_16
}

