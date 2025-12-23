
import pandas as pd

# Prepare a summary table for the metrics extracted
summary_metrics = {
    'Metric': [
        'Final epidemic size (R at max time)',
        'Final epidemic fraction of N',
        'Peak number of Infectious (I)',
        'Time of peak Infectious (days)',
        'Epidemic duration (days)',
        'Early exponential growth rate of I (/day)',
        'Empirical doubling time of I (days)',
        'Theoretical R0 (ER network)',
        'Initial Exposed seeds',
        'Initial Infectious seeds'
    ],
    'Value': [
        final_data['R'],
        final_data['R'] / N,
        peak_I,
        peak_time,
        epidemic_duration,
        growth_rate,
        doubling_time,
        R0,
        initial_seeds['E'],
        initial_seeds['I']
    ],
    'Unit': [
        'individuals',
        'fraction of N',
        'individuals',
        'days',
        'days',
        'per day',
        'days',
        'dimensionless',
        'individuals',
        'individuals'
    ]
}

summary_df = pd.DataFrame(summary_metrics)

# Save summary table to output directory
summary_file = os.path.join(o_dir, 'SEIR_simulation_metrics_summary.csv')
summary_df.to_csv(summary_file, index=False)

summary_file, summary_df