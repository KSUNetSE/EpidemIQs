
import pandas as pd

# We have learned the structure of typical single-run results files
# Columns: time, S, S_90ci_lower, S_90ci_upper, I, I_90ci_lower, I_90ci_upper, R, R_90ci_lower, R_90ci_upper
# Let's write a function that extracts the metrics from a single run file

def extract_sir_metrics(df):
    N_total = df[['S', 'I', 'R']].iloc[0].sum()  # total population (usually constant)
    
    # Final epidemic size: R at last timestep
    final_R = df['R'].iloc[-1]
    final_R_frac = final_R / N_total
    
    # Peak infection count and time
    peak_I = df['I'].max()
    peak_idx = df['I'].idxmax()
    peak_time = df['time'].iloc[peak_idx]

    # Epidemic duration: from first nonzero I to last nonzero I
    infected_nonzero = df[df['I'] > 0]
    duration = infected_nonzero['time'].iloc[-1] - infected_nonzero['time'].iloc[0] if len(infected_nonzero) > 0 else 0
    
    # No vaccination info in these files, so None
    vaccinated_frac = None
    vaccinated_num = None
    
    return {
        'N_total': N_total,
        'final_R': final_R,
        'final_R_frac': final_R_frac,
        'peak_I': peak_I,
        'peak_time': peak_time,
        'duration': duration,
        'vaccinated_num': vaccinated_num,
        'vaccinated_frac': vaccinated_frac
    }