
# Check if dataset contains columns of individual runs or final size per run
dataset_columns = list(data.columns)
run_col_exists = any(['run' in col.lower() or 'simulation' in col.lower() for col in dataset_columns])

unique_run_counts = None
if run_col_exists:
    run_col = [col for col in dataset_columns if 'run' in col.lower() or 'simulation' in col.lower()][0]
    unique_run_counts = data[run_col].nunique()

run_col_exists, unique_run_counts, dataset_columns