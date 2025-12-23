
results_all = {}

for fp in [
    'output/results-11.csv',
    'output/results-12.csv',
    'output/results-13.csv',
    'output/results-14.csv',
    'output/results-15.csv',
    'output/results-16.csv'
]:
    df_temp = pd.read_csv(fp)
    res = analyze_file(df_temp)
    results_all[fp] = res

results_all