
# We will now apply this function to all single-run SIR results files (excluding attackrate summary csvs)

filepaths_single = [
    'output/results-11.csv', 'output/results-12.csv', 'output/results-13.csv', 'output/results-14.csv',
    'output/results-15.csv', 'output/results-16.csv', 'output/results-21.csv', 'output/results-31.csv',
    'output/results-41.csv', 'output/results-42.csv', 'output/results-43.csv', 'output/results-44.csv',
    'output/results-45.csv', 'output/results-46.csv'
]

summary_results = {}

for fp in filepaths_single:
    df = pd.read_csv(fp)
    metrics = extract_sir_metrics(df)
    summary_results[fp] = metrics

summary_results