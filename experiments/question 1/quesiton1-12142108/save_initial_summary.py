
import os

# Ensure output directory exists
dir_path = os.path.join(os.getcwd(), "output")
os.makedirs(dir_path, exist_ok=True)

# Save summary metrics to CSV
summary_metrics = {
    "Metric": ["Final R (absolute)", "Final R (fraction)", "Max I", "Peak day", "Epidemic start day", "Epidemic end day", "Epidemic duration (days)", "Initial E", "Initial I", "Early growth rate r", "Doubling time (days)", "Theoretical R0"],
    "Value": [final_R_abs, final_R_frac, max_I, peak_day, epidemic_start, epidemic_end, epidemic_duration, initial_E, initial_I, np.nan, np.nan, np.nan]
}

summary_df = pd.DataFrame(summary_metrics)

csv_path = os.path.join(dir_path, "summary_metrics.csv")
summary_df.to_csv(csv_path, index=False)
csv_path