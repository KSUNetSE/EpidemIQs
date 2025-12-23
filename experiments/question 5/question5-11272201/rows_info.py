
# Additionally checking number of rows in each file to see run counts and length
rows_info = {fp: len(pd.read_csv(fp)) for fp in filepaths}
rows_info