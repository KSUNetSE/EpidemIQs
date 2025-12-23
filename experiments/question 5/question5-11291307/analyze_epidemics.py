
import numpy as np

def analyze_epidemic(df, total_population, vaccinated_population):
    # Non-vaccinated population
    non_vaccinated_population = total_population - vaccinated_population
    
    # Final epidemic size fraction: (final R - initial R) / non_vaccinated_population
    final_epidemic_sizes = []
    peak_infected_numbers = []
    times_to_peak = []
    epidemic_durations = []
    minor_outbreak_count = 0
    total_realizations = len(np.unique(df['realization'])) if 'realization' in df.columns else 1
    
    # In case we have single realization data only (one curve)
    if total_realizations == 1:
        initial_R = df['R'].iloc[0]
        final_R = df['R'].iloc[-1]
        final_epidemic_size = (final_R - initial_R) / non_vaccinated_population
        final_epidemic_sizes.append(final_epidemic_size)
        
        peak_I = df['I'].max()
        peak_infected_numbers.append(peak_I)
        
        time_to_peak = df['time'][df['I'].idxmax()]
        times_to_peak.append(time_to_peak)
        
        duration = df['time'].iloc[-1]  # assuming last time point is when I=0
        epidemic_durations.append(duration)
        
        minor_outbreak_count = int(final_epidemic_size <= 2 / non_vaccinated_population)
        
    else:
        for realization in np.unique(df['realization']):
            realization_data = df[df['realization'] == realization]
            initial_R = realization_data['R'].iloc[0]
            final_R = realization_data['R'].iloc[-1]
            final_epidemic_size = (final_R - initial_R) / non_vaccinated_population
            final_epidemic_sizes.append(final_epidemic_size)
            
            peak_I = realization_data['I'].max()
            peak_infected_numbers.append(peak_I)
            
            time_to_peak = realization_data['time'][realization_data['I'].idxmax()]
            times_to_peak.append(time_to_peak)
            
            # duration: time from first nonzero I to last nonzero I
            nonzero_I_times = realization_data.loc[realization_data['I'] > 0, 'time']
            if len(nonzero_I_times) > 0:
                duration = nonzero_I_times.iloc[-1] - nonzero_I_times.iloc[0]
            else:
                duration = 0
            epidemic_durations.append(duration)
            
            if final_epidemic_size <= 2 / non_vaccinated_population:
                minor_outbreak_count += 1

    def confidence_interval(data, ci=0.9):
        lower = np.percentile(data, (1-ci)/2*100)
        upper = np.percentile(data, (1+(ci))/2*100)
        mean_or_median = np.mean(data)
        median = np.median(data)
        return (mean_or_median, median, lower, upper)

    results = {
        'final_epidemic_size': confidence_interval(final_epidemic_sizes),
        'peak_infected_numbers': confidence_interval(peak_infected_numbers),
        'times_to_peak': confidence_interval(times_to_peak),
        'epidemic_durations': confidence_interval(epidemic_durations),
        'minor_outbreak_count': minor_outbreak_count,
        'minor_outbreak_fraction': minor_outbreak_count / total_realizations,
        'total_realizations': total_realizations
    }
    
    # Additional outbreak statistics
    results['final_epidemic_size_std'] = np.std(final_epidemic_sizes)
    results['peak_infected_std'] = np.std(peak_infected_numbers)
    results['epidemic_duration_std'] = np.std(epidemic_durations)

    # Initial t=0 counts for vaccinated and susceptible
    initial_S = df['S'].iloc[0]
    initial_R = df['R'].iloc[0]
    initial_I = df['I'].iloc[0]

    results['initial_S'] = initial_S
    results['initial_R'] = initial_R
    results['initial_I'] = initial_I
    results['vaccinated_at_t0'] = vaccinated_population

    return results

# From previous inspection
# For results-11.csv
# total population ~ initial S + initial R (since I is very small initially)
# initial R ~ 7500
# initial S ~ 2500 (vaccinated 75% coverage -> vaccinated ~ 7500, non-vaccinated ~ 2500 in previous inspection)

# For results-21.csv
# total population ~ initial S + initial R
# initial S ~ 8884, initial R ~ 1115 (vaccinated covering high degrees only)

# Run the analysis on both datasets
# Note: The data does not have an explicit 'realization' column so it appears it's aggregated/averaged results across 100 runs

# Aggregate data: Since no per realization data, we treat as single realization aggregated data
# Thus we will report just the single values rather than CIs across realizations

result_11 = analyze_epidemic(results_11, total_population=10000, vaccinated_population=7500)
result_21 = analyze_epidemic(results_21, total_population=10000, vaccinated_population=1115)

result_11, result_21