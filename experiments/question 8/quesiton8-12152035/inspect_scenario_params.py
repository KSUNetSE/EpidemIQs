
import pandas as pd

# Load the scenario parameter map
param_file = '/Users/hosseinsamaei/phd/epidemiqs/output/scenario_parameter_map.csv'
df_params = pd.read_csv(param_file)

# Inspect the dataframe structure
param_overview = {
    'columns': df_params.columns.tolist(),
    'head': df_params.head(5)
}
param_overview