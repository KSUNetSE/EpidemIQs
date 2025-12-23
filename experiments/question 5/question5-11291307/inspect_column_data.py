
# Check for unique values in important columns that might represent time and states (S, I, R, V for vaccinated maybe)
unique_11_cols = {col: results_11[col].unique()[:5] for col in results_11.columns if results_11[col].dtype != 'object'}
unique_21_cols = {col: results_21[col].unique()[:5] for col in results_21.columns if results_21[col].dtype != 'object'}

# Check column data types
dtypes_11 = results_11.dtypes
dtypes_21 = results_21.dtypes