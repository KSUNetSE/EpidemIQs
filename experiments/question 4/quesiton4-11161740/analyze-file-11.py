
def analyze_file(df):
    # Extract time vector
    time = df['time'].values
    total_time = time[-1]
    n = len(time)
    last_20pct_index = int(n * 0.8)  # Index to start last 20%
    last_5pct_index = int(n * 0.95)  # Index to start last 5%
    
    # Steady-state prevalence mean and 90% CI over last 20%
    steady_I1_mean = df['I1_mean'][last_20pct_index:].mean()
    steady_I1_lower = df['I1_90ci_lower'][last_20pct_index:].mean()
    steady_I1_upper = df['I1_90ci_upper'][last_20pct_index:].mean()
    steady_I2_mean = df['I2_mean'][last_20pct_index:].mean()
    steady_I2_lower = df['I2_90ci_lower'][last_20pct_index:].mean()
    steady_I2_upper = df['I2_90ci_upper'][last_20pct_index:].mean()
    
    # Peak values for I1 and I2
    peak_I1 = df['I1_mean'].max()
    peak_I2 = df['I2_mean'].max()

    # Time to steady state (stabilization) defined by relative change < 1e-3 over last 5% window
    def time_to_steady(I):
        for i in range(last_5pct_index, n):
            window = df[I][i - int(n*0.05):i]
            rel_change = np.abs(window.pct_change().dropna())
            if rel_change.max() < 1e-3:
                return df['time'].iloc[i]
        return df['time'].iloc[-1]  # if never stabilized

    time_steady_I1 = time_to_steady('I1_mean')
    time_steady_I2 = time_to_steady('I2_mean')

    # Number/fraction of times where I1 or I2 extinctions occur (<1% prevalence)
    extinct_I1_frac = (df['I1_mean'] < 0.01).mean()
    extinct_I2_frac = (df['I2_mean'] < 0.01).mean()

    # Determine outcome
    coexistence = (steady_I1_mean > 0.01) and (steady_I2_mean > 0.01)
    dominance_1 = (steady_I1_mean > 0.01) and (steady_I2_mean <= 0.01)
    dominance_2 = (steady_I2_mean > 0.01) and (steady_I1_mean <= 0.01)
    bistable_extinct = not (coexistence or dominance_1 or dominance_2)

    if coexistence:
        outcome = 'coexistence'
    elif dominance_1:
        outcome = 'dominance_1'
    elif dominance_2:
        outcome = 'dominance_2'
    else:
        outcome = 'bistable_or_extinct'

    return {
        'steady_I1_mean': steady_I1_mean,
        'steady_I1_CI': (steady_I1_lower, steady_I1_upper),
        'steady_I2_mean': steady_I2_mean,
        'steady_I2_CI': (steady_I2_lower, steady_I2_upper),
        'peak_I1': peak_I1,
        'peak_I2': peak_I2,
        'time_steady_I1': time_steady_I1,
        'time_steady_I2': time_steady_I2,
        'extinct_I1_frac': extinct_I1_frac,
        'extinct_I2_frac': extinct_I2_frac,
        'outcome': outcome
    }

results_11 = analyze_file(df)