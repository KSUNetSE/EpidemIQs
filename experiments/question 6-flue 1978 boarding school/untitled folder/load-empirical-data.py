
import pandas as pd

# Load empirical data from Excel
empirical_file_path = 'cases_data.xlsx'
empirical_data = pd.read_excel(empirical_file_path)

# Preview empirical data
empirical_data.head(), empirical_data.columns, empirical_data.shape