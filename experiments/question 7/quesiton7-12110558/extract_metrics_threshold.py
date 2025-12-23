
# Recalculate summary based on >50% final_failed as cascade
summary_threshold = []

for path, df in dataframes.items():
    if df is not None and 'final_failed' in df.columns:
        total_runs = len(df)
        if total_runs == 0:
            cascade_prob = 'NA'
            mean_failed = 'NA'
            std_failed = 'NA'
        else:
            # Use final_failed > 0.5 to determine cascade
            num_global_cascades = (df['final_failed'] > 0.5).sum()
            cascade_prob = num_global_cascades / total_runs
            mean_failed = df['final_failed'].mean()
            std_failed = df['final_failed'].std()
    else:
        total_runs = 'NA'
        cascade_prob = 'NA'
        mean_failed = 'NA'
        std_failed = 'NA'

    summary_threshold.append((scenarios[path]['kc'], scenarios[path]['seeding'], total_runs, cascade_prob, mean_failed, std_failed))

summary_threshold