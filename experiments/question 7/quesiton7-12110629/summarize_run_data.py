
# Since results_summary does not include mean steps to convergence, we need to calculate it manually from the runs data
# Extract mean, cascade probability and mean steps to convergence for both scenarios from runs data

def summarize_runs(df):
    num_runs = df['run'].nunique()
    final_failed_fraction = df['frac_failed']
    systemic_failures = df['systemic_failure']
    steps_to_convergence = df['tsteps']
    
    cascade_prob = systemic_failures.mean()
    mean_failed = final_failed_fraction.mean()
    mean_steps = steps_to_convergence.mean()
    
    return {
        'num_runs': num_runs,
        'cascade_probability': cascade_prob,
        'mean_final_failed_fraction': mean_failed,
        'mean_steps_to_convergence': mean_steps,
        'details': df
    }

summary_10 = summarize_runs(results_10)
summary_11 = summarize_runs(results_11)