
import pandas as pd
# Collate metrics from the two runs
# Simulation 1: random seeding
metrics1 = {'Peak Infected': 3, 'Peak Time (days)': 0.0, 'Final Epidemic Size (R at end)': 3, 'Epidemic Duration (days)': 36.95, 'Doubling Time (days)': None}
# Simulation 2: hub seeding
metrics2 = {'Peak Infected': 16, 'Peak Time (days)': 56.67, 'Final Epidemic Size (R at end)': 41, 'Epidemic Duration (days)': 120.33, 'Doubling Time (days)': 8.96}

metrics_df = pd.DataFrame([
    {'Seeding': 'Random (3 infected)', **metrics1},
    {'Seeding': 'Hub (top 3 degree)', **metrics2},
])
metrics_df.to_csv('output/metrics_table.csv', index=False)
metrics_df