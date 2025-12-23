
# Load the WS network simulation time series CSV to examine structure and time series
file_path_ws = '/Users/hosseinsamaei/phd/epidemiqs/output/results-11.csv'
ws_data = pd.read_csv(file_path_ws)

# Check dataset structure and first few rows
head_ws_data = ws_data.head()
columns_ws = ws_data.columns.tolist()
# Also check shape to get sense of data size
shape_ws = ws_data.shape

head_ws_data, columns_ws, shape_ws