
import numpy as np

summary = []

scenarios = {
    'output/results-11.csv': {'kc': 0.90, 'seeding': 'core'},
    'output/results-21.csv': {'kc': 0.90, 'seeding': 'periphery'},
    'output/results-14.csv': {'kc': 1.00, 'seeding': 'core'},
    'output/results-24.csv': {'kc': 1.00, 'seeding': 'periphery'},
}

for path, df in dataframes.items():
    runs = len(df) if df is not None else 'NA'
    if runs == 0:
        cascade_prob = 'NA'
        mean_failed = 'NA'
        std_failed = 'NA'
        summary.append((scenarios[path]['kc'], scenarios[path]['seeding'], runs, cascade_prob, mean_failed, std_failed))
        continue
    # Check for missing data
    if 'final_failed' in df.columns and 'is_global_cascade' in df.columns:
        total_runs = len(df)
        if total_runs == 0:
            cascade_prob = 'NA'
            mean_failed = 'NA'
            std_failed = 'NA'
        else:
            num_global_cascades = df['is_global_cascade'].sum()
            cascade_prob = num_global_cascades / total_runs
            mean_failed = df['final_failed'].mean()
            std_failed = df['final_failed'].std()
    else:
        total_runs = 'NA'
        cascade_prob = 'NA'
        mean_failed = 'NA'
        std_failed = 'NA'
    summary.append((scenarios[path]['kc'], scenarios[path]['seeding'], total_runs, cascade_prob, mean_failed, std_failed))

summary