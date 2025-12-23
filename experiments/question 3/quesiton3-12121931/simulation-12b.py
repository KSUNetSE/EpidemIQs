
# Examine the returns: if there is no column 't', check other possible names.
# Print column names and a preview row-by-row for inferring the correct one for time, node1, node2.
# If names are ambiguous, provide all three names for mapping.
import pandas as pd

table_path = '/Users/hosseinsamaei/phd/epidemiqs/output/temporal-edge-table-n1000-alpha01-m2-T1000.csv'
edges_df = pd.read_csv(table_path)
list(edges_df.columns), edges_df.head(10).to_dict(orient='records')