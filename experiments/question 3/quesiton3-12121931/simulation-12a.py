
# Reason about error: Edge-list file does not contain a column 't' (timestep). Let's inspect its columns.
import pandas as pd
table_path = '/Users/hosseinsamaei/phd/epidemiqs/output/temporal-edge-table-n1000-alpha01-m2-T1000.csv'
edges_df = pd.read_csv(table_path)
columns = list(edges_df.columns)
rows_preview = edges_df.head(5)
(rows_preview, columns)