
file_path_detailed = '/Users/hosseinsamaei/phd/epidemiqs/output/results-21.csv'
detailed_data = pd.read_csv(file_path_detailed)

columns_detailed = detailed_data.columns.tolist()
data_detailed_head = detailed_data.head()
data_detailed_shape = detailed_data.shape

columns_detailed, data_detailed_head, data_detailed_shape