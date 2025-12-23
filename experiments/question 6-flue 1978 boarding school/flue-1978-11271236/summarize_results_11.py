
# Redefining dictionary with metrics_11 from previous step for consistency
metrics_11_final = {
    'Final attack rate (%)': 100.0,
    'Peak Bed Occupancy (max B)': 375.565,
    'Peak Bed Timing (day)': 2.9673312856204492,
    'Epidemic Duration (days)': 34.14532475509054,
    'Mean Duration in B (days)': 5.350538337328903,
    'Final C count': 763.0
}

# Compile all metrics including metrics_11
all_metrics = {'results-11': metrics_11_final}
all_metrics.update(metrics_all)

# Verify consistency for results-11 with previously extracted metrics
validation_11 = all_metrics['results-11']
validation_11