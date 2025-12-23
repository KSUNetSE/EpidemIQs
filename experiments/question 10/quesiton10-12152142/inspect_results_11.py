
# Load and inspect results-11.csv
file_path_detailed = os.path.join(directory, 'results-11.csv')
data_detailed = pd.read_csv(file_path_detailed)

columns_detailed = data_detailed.columns.tolist()
sample_detailed = data_detailed.head()
data_detailed_shape = data_detailed.shape