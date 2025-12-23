
# Analysis of epidemic SIR curves from csv
data_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
import pandas as pd
# Load
results = pd.read_csv(data_path)
# Peak infected: max I, when
peak_infect = int(results['I'].max())
peak_time = float(results['time'][results['I'].idxmax()])
# Final epidemic size (total R at end)
final_size = int(results['R'].iloc[-1])
# Total duration: last time infected >0
infected_nonzero = results[results['I'] > 0]
end_time = float(infected_nonzero['time'].iloc[-1])
# Initial susceptible
init_sus = int(results['S'].iloc[0])
# Description of curve shape and phases
def get_curve_shape(series):
    diff = series.diff().fillna(0)
    # Classify phase: rise, steady, decline
    rise = (diff > 0).sum()
    fall = (diff < 0).sum()
    return {'rise_count': int(rise), 'fall_count': int(fall)}
curve_I = get_curve_shape(results['I'])
curve_S = get_curve_shape(results['S'])
curve_R = get_curve_shape(results['R'])
# Assemble metrics and shapes
analysis = {
    'peak_infect': peak_infect,
    'peak_time': peak_time,
    'final_epidemic_size': final_size,
    'epidemic_duration': end_time,
    'initial_susceptible': init_sus,
    'curve_I_rise': curve_I['rise_count'],
    'curve_I_fall': curve_I['fall_count'],
    'curve_S_rise': curve_S['rise_count'],
    'curve_S_fall': curve_S['fall_count'],
    'curve_R_rise': curve_R['rise_count'],
    'curve_R_fall': curve_R['fall_count']
}
analysis