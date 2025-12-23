
filepaths_attack = ['output/results-1attackrates.csv', 'output/results-4attackrates.csv']

attackrate_summaries = {}

for fp in filepaths_attack:
    df = pd.read_csv(fp)
    attackrate_summaries[fp] = extract_attackrate_sweep(df)

attackrate_summaries