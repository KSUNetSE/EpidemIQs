
import pandas as pd

# Load the simulation results CSV to inspect its structure
file_path = 'output/results-22.csv'
data = pd.read_csv(file_path)
data.head(), data.columns, data.shape
import pandas as pd

# Load the data to understand its structure
file_path = 'output/results-12.csv'
data = pd.read_csv(file_path)
data.head(), data.columns.tolist()
import pandas as pd

# Load a small sample of the file to inspect its structure
df_sample = pd.read_csv('output/results-21.csv', nrows=10)
df_sample.head()
import pandas as pd

# Load the data to examine its contents
file_path = 'output/results-11.csv'
data = pd.read_csv(file_path)
data.head(), data.columns, data.shape
import numpy as np

# Extract unique times and states count
unique_times = np.sort(data['time'].unique())

# Preliminary check: Number of unique time points
num_unique_times = len(unique_times)

num_runs = 75  # We know there are 75 runs
num_time_points = num_unique_times // num_runs

num_runs, num_time_points
# The data columns are time, S, I, R which represent susceptible, infected, and recovered populations over time respectively.
# It appears to be a time-series from one run. Since we want metrics aggregated over 75 runs, 
# we should check how multiple runs are represented in the data.

# Read more rows to see if there is a run index or multiple runs are stacked in some way
df_more = pd.read_csv('output/results-21.csv', nrows=100)
df_more.tail(20)
# The data indicates columns: time, Susceptible (S), Infected (I), and Recovered (R). It appears to be from one run.
# We need to confirm if the file contains multiple runs or if multiple files exist for 75 runs.

# Count unique time values and unique runs if any
num_unique_times = data['time'].nunique()
num_total_rows = data.shape[0]

# This check to understand if data for multiple runs are stacked or if we have 1 run only
ratio = num_total_rows / num_unique_times
ratio
# Check if there is an explicit run identifier by scanning first 1000 lines
df_ids = pd.read_csv('output/results-21.csv', nrows=1000)
unique_times = df_ids['time'].unique()
len(unique_times)
# Reasoning on data structure
# It appears to be a time series with columns for time, S, I, and R
# We need to confirm if this is for multiple runs concatenated or just one run
# Examine basic statistics and unique time points
unique_times = data['time'].unique()
min_time = min(unique_times)
max_time = max(unique_times)
num_unique_times = len(unique_times)
unique_S = data['S'].unique()
unique_I = data['I'].unique()
unique_R = data['R'].unique()

print(f"Unique time points: {num_unique_times}, Time range: {min_time} to {max_time}")
print(f"Unique S values count: {len(unique_S)}")
print(f"Unique I values count: {len(unique_I)}")
print(f"Unique R values count: {len(unique_R)}")
# Check total number of rows in the data
num_rows = data.shape[0]
num_rows
# Since ratio is 1.0, data appears to be from a single run, but user expects mean or median over 75 runs.
# We need to confirm if there is other files or file contains separate data for multiple runs in another dimensionality.

# Let's check the file directory and see if more result files exist for other runs
import os

files_in_output = os.listdir('output')
files_in_output
# Check if the data corresponds to multiple runs or if these are single run results
# We expect to see a 'run' column or similar to distinguish runs
run_column_exists = 'run' in data.columns
run_column_exists
# We know there are 75 runs, check the number of unique time points
num_unique_times = len(np.unique(data['time']))

num_unique_times, num_rows / 75
# There is no explicit run identifier. Possibly the data is concatenated sequentially for 75 runs.
# Let's check total number of rows and try to split by number of runs (75) to see equal run lengths

df_full = pd.read_csv('output/results-21.csv')
num_rows = df_full.shape[0]
num_runs = 75
rows_per_run = num_rows // num_runs
num_rows, rows_per_run
# Check for files that look like results from runs, e.g., results-*.csv
import re

result_files = [f for f in files_in_output if re.match(r'results-\d+\.csv', f)]
result_files_sorted = sorted(result_files)
result_files_sorted
# Since no explicit run column, let's check if the data has 75 equal segments corresponding to 75 runs
num_rows = data.shape[0]
runs_guess = 75
rows_per_run = num_rows // runs_guess

# Check if rows_per_run * runs_guess == num_rows, to confirm if even segments
exact_segments = (rows_per_run * runs_guess == num_rows)
rows_per_run, exact_segments
# Check if splitting the data evenly by rows_per_run aligns with expected time values
runs = []
for i in range(num_runs):
    runs.append(df_full.iloc[i*rows_per_run:(i+1)*rows_per_run])
runs[0].head()
# Since number of unique time points equals num_rows, data is stored for all 75 runs at each time point, likely the runs are concatenated vertically
# Extract number of time points by checking max time for each run (assuming time starts at 0 for each run)

# Check if there are repeated time 0, expected to be 75 (one for each run)
num_time0 = (data['time'] == 0).sum()
num_time0
# Checking if there are repeated time values: expect time series per run
repeated_times = data['time'].duplicated().any()
repeated_times

# Since only one time 0, data is likely from a single run or aggregated over runs, not 75 separate trajectories
# Check min and max time
min_time = data['time'].min()
max_time = data['time'].max()

(min_time, max_time)
# Check if time values are evenly spaced and if any discontinuities exist
# Also check range and difference between last and first time
import numpy as np

time_diff = np.diff(data['time'].values)
median_diff = np.median(time_diff)
min_diff = np.min(time_diff)
max_diff = np.max(time_diff)
only_small_diff = (min_diff > 0) and (max_diff - min_diff < 1e-5)
time_diff[:10], median_diff, min_diff, max_diff, only_small_diff
# Summary statistics for I (Infected)
I_min = data['I'].min()
I_max = data['I'].max()
I_final = data['I'].iloc[-1]

# Summary statistics for S (Susceptible) and R (Recovered)
S_initial = data['S'].iloc[0]
S_final = data['S'].iloc[-1]
R_initial = data['R'].iloc[0]
R_final = data['R'].iloc[-1]

I_min, I_max, I_final, S_initial, S_final, R_initial, R_final
# Since there is time gaps, maybe time series is concatenated for different runs
# Check the repeated S and I and R values pattern at time=0 to find segmentation indexes
initial_idx_times = data['time'] < 0.1
initial_idx = data[initial_idx_times].index
initial_idx.tolist()
# Check occurrence of minimum time which should be 0 for each run start
min_times_index = data.index[data['time'] == 0]
min_times_index.tolist()
# We have combined data from 4 runs. To approach 75 runs mean/median, we should check if these 4 files correspond to multiple runs each or if we are limited to these.
# For now, we'll calculate the requested metrics averaged over these runs.

# Extract number of runs
num_runs = len(unique_runs)

# Summary statistics: mean and median for each time point over runs
# First, verify if the time points align across runs
unique_times_per_run = {run: combined_data[combined_data['run']==run]['time'].unique() for run in unique_runs}

# Check if all time arrays are equal (same timestamps for each run) for aggregation by time
all_times_equal = all(np.array_equal(unique_times_per_run[unique_runs[0]], unique_times_per_run[run]) for run in unique_runs)

all_times_equal
# Extract epidemic duration: time until I(t) remains zero to the end
# Find the last time point at which I>0
last_inf_time = data.loc[data['I'] > 0, 'time'].max()

# Peak prevalence: max I and occurrence time
peak_prevalence = data['I'].max()
peak_time = data.loc[data['I'] == peak_prevalence, 'time'].values[0]

# Final epidemic size = final R or N - final S (population assumed constant)
final_epidemic_size = data['R'].iloc[-1]
population = S_initial + I_max + R_initial  # Check initial population
complement_epidemic_size = population - data['S'].iloc[-1]

last_inf_time, peak_prevalence, peak_time, final_epidemic_size, complement_epidemic_size
# Look for repeated 0 times for identifying different runs
# Since only one 0 time, data seems to be for a single run or average over runs
# Let's analyze the data for requested metrics on single run or average
# Check if I(t) goes to zero and stays zero (epidemic duration)
nonzero_I = data[data['I'] > 0]
last_nonzero_time = nonzero_I['time'].iloc[-1] if not nonzero_I.empty else 0
last_time = data['time'].iloc[-1]

# Peak prevalence
peak_I = data['I'].max()
peak_time = data.loc[data['I'].idxmax(), 'time']

# Final epidemic size (R at last time)
final_R = data['R'].iloc[-1]

# Initial and final S
initial_S = data['S'].iloc[0]
final_S = data['S'].iloc[-1]

last_nonzero_time, last_time, peak_I, peak_time, final_R, initial_S, final_S
# Since times are not equal across runs, we interpolate to a common time grid for aggregation
common_time_grid = np.linspace(combined_data['time'].min(), combined_data['time'].max(), 1000)  # 1000 points

# Interpolate S, I, R for each run on common time grid
interpolated_runs = []

for run in unique_runs:
    subset = combined_data[combined_data['run'] == run]
    interp_S = np.interp(common_time_grid, subset['time'], subset['S'])
    interp_I = np.interp(common_time_grid, subset['time'], subset['I'])
    interp_R = np.interp(common_time_grid, subset['time'], subset['R'])
    df_interp = pd.DataFrame({'time': common_time_grid, 'S': interp_S, 'I': interp_I, 'R': interp_R, 'run': run})
    interpolated_runs.append(df_interp)

interpolated_data = pd.concat(interpolated_runs, ignore_index=True)

# Calculate mean and median over runs for each time point
mean_time_series = interpolated_data.groupby('time').agg({'S':'mean', 'I':'mean', 'R':'mean'}).reset_index()
median_time_series = interpolated_data.groupby('time').agg({'S':'median', 'I':'median', 'R':'median'}).reset_index()

mean_time_series.head(), median_time_series.head()
# Calculate effective reproduction number Re(t) = R0 * S(t)/N
# Estimate R0 during initial growth from doubling time is not straightforward with R0 unknown
# Instead, calculate doubling time during initial exponential growth phase of I(t)

from scipy.optimize import curve_fit
import numpy as np

def exp_growth(t, r, I0):
    return I0 * np.exp(r * t)

# Use early times when I is increasing (I> initial infected and before peak prevalence)
early_phase = data[(data['time'] <= peak_time) & (data['I'] > data['I'].iloc[0])]

# Fit exponential to infected during early phase
popt, pcov = curve_fit(exp_growth, early_phase['time'], early_phase['I'], p0=(0.1, early_phase['I'].iloc[0]), maxfev=10000)

growth_rate = popt[0]  # exponential growth rate
initial_infected = popt[1]

doubling_time = np.log(2) / growth_rate if growth_rate > 0 else np.inf

data['Re_t'] = (peak_prevalence + 1) * data['S'] / population  # Proxy using peak prevalence +1 as R0 approx

# Time where Re(t) drops below 1
try:
    time_Re_below_1 = data.loc[data['Re_t'] < 1, 'time'].values[0]
except IndexError:
    time_Re_below_1 = None

growth_rate, doubling_time, time_Re_below_1
# Doubling time during initial growth phase
# We consider initial phase with I(t) < peak_I / 2 and I > 0
initial_phase = data[(data['I'] > 0) & (data['I'] <= peak_I / 2)]
# Use the exponential growth approximation: I(t) = I0 * exp(r*t), doubling time = ln(2)/r
# Fit a line to log(I) vs t for initial phase
import numpy as np

# Filter initial phase where I is strictly positive and less than half peak
initial_phase = initial_phase[initial_phase['I'] > 0]

t_vals = initial_phase['time'].values
log_I = np.log(initial_phase['I'].values)
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(t_vals, log_I)
doubling_time = np.log(2) / slope if slope > 0 else np.nan
doubling_time
# Calculate outbreak probability: fraction of runs with nonzero final size
# However, we have only one run or mean/median data
# Let's look at the final R value to guess if outbreak occurred
outbreak_occurred = final_epidemic_size > 0

outbreak_occurred
# Review initial_phase data to understand why doubling_time was nan
initial_phase[['time', 'I']], initial_phase['I'].min(), initial_phase['I'].max()
# Since doubling time was NaN, maybe slope was negative or too noisy
# Filter to the very initial increasing phase (e.g., until peak / 10) for stable exponential growth
initial_phase_strict = data[(data['I'] > 0) & (data['I'] <= peak_I / 10)]
t_vals = initial_phase_strict['time'].values
log_I = np.log(initial_phase_strict['I'].values)
slope, intercept, r_value, p_value, std_err = linregress(t_vals, log_I)
doubling_time_strict = np.log(2) / slope if slope > 0 else np.nan
doubling_time_strict, slope, r_value**2
# We got negative slope, maybe an initial outbreak was not sustained or data has too much fluctuations
# Let's visualize initial I(t) to see the trend
import matplotlib.pyplot as plt
plt.plot(data['time'], data['I'])
plt.xlabel('Time (days)')
plt.ylabel('Infected individuals')
plt.title('Infected over time - full simulation')
plt.show()
# We can calculate Re(t) = R0 * S(t) / N, we need to estimate R0 using initial data
# Approximate total population size N
N = initial_S + data['I'].iloc[0] + data['R'].iloc[0]

# Calculate effective reproduction number Re(t) using an estimated R0 value
# To estimate R0, assume initial exponential growth: r = slope, and R0 approx 1 + r * D (infectious period D assumed as 5 for demo)
# But slope was negative, so we can't estimate R0 this way
# Let's just calculate Re(t) = R0 * S(t) / N and watch when it drops below 1
# We have no direct R0, but given initial infected increase, R0 > 1
# So instead, find time when Re(t) drops below 1, for Re(t) = R0 * S(t) / N, that is proportional to S(t)

# We don't know R0 separately, but we can find when S(t)/N < 1/R0
# Let's define t_cut = time when S(t) < N / R0

# Since we don't have R0, let's calculate S(t)/N over time

S_over_N = data['S'] / N

# Assume rough R0 from initial infected increase (peak_I initial)?
# Or consider R0 as max Re(t) during initial phase with high I
# We can approximate R0 as peak I divided by initial I, but that is not exact

# Without direct R0 value, let's define Re_t as R0 * S(t)/ N and find the time when it drops below 1 numerically
# We will calculate Re(t) as proportional to S(t) and find when it's below threshold relative to initial value

# Define a threshold as fraction of initial S over N to mimic Re dropping below 1
threshold_fraction = 1/5  # arbitrary, because no direct R0 provided
below_thresh_idx = S_over_N < threshold_fraction
if below_thresh_idx.any():
    time_below_re1 = data['time'][below_thresh_idx].iloc[0]
else:
    time_below_re1 = np.nan

time_below_re1
# Check for outbreak probability: fraction of runs with non-zero final size
# Since data appears to be single run or mean over runs, this cannot be derived directly
# We can ask if final_R is greater than zero
outbreak_occurred = final_R > 0
outbreak_probability = 1.0 if outbreak_occurred else 0.0
outbreak_probability
# Extract requested metrics based on mean or median time series

# 1) Epidemic Duration (time until I(t) remains zero)
# Find last time I(t) > 0

epidemic_duration_mean = mean_time_series[mean_time_series['I'] > 0]['time'].max()
epidemic_duration_median = median_time_series[median_time_series['I'] > 0]['time'].max()

# 2) Peak Prevalence (max I(t) value and occurrence time)
peak_mean = mean_time_series.loc[mean_time_series['I'].idxmax()][['time', 'I']]
peak_median = median_time_series.loc[median_time_series['I'].idxmax()][['time', 'I']]

# 3) Final Epidemic Size (final R or N-S(infinity))
# Final time point assumed last in the series
final_R_mean = mean_time_series['R'].iloc[-1]
final_R_median = median_time_series['R'].iloc[-1]
final_S_mean = mean_time_series['S'].iloc[-1]
final_S_median = median_time_series['S'].iloc[-1]

# Total population N assumed constant and can be calculated (N = S + I + R at t=0)
N_mean = mean_time_series.loc[0, 'S'] + mean_time_series.loc[0, 'I'] + mean_time_series.loc[0, 'R']
N_median = median_time_series.loc[0, 'S'] + median_time_series.loc[0, 'I'] + median_time_series.loc[0, 'R']

# 4) Doubling time during initial growth (if outbreak occurs, using initial infection growth phase)
# Estimate doubling time from exponential growth phase of I(t)
# Use time points where I(t) is increasing above initial to peak

initial_growth_mean = mean_time_series[mean_time_series['I'] > mean_time_series['I'].iloc[0]]
initial_growth_median = median_time_series[median_time_series['I'] > median_time_series['I'].iloc[0]]

# Ensure enough data points for growth
from scipy.stats import linregress
import numpy as np

doubling_time_mean = None
if not initial_growth_mean.empty:
    log_I = np.log(initial_growth_mean['I'])
    slope, intercept, r_value, p_value, std_err = linregress(initial_growth_mean['time'], log_I)
    if slope > 0:
        doubling_time_mean = np.log(2) / slope

doubling_time_median = None
if not initial_growth_median.empty:
    log_I_med = np.log(initial_growth_median['I'])
    slope_m, intercept_m, r_value_m, p_value_m, std_err_m = linregress(initial_growth_median['time'], log_I_med)
    if slope_m > 0:
        doubling_time_median = np.log(2) / slope_m

# 5) Time at which effective Re(t) = R0·S(t)/N drops below 1 (if outbreak occurs)
# We don't have explicit R0, but we can estimate it from initial growth rate.
# Approximation: R0 = 1 + growth_rate * infectious_period
# Since we don't have infectious_period here, we cannot directly compute R0 but we can calculate Re(t) = R0 * S(t) / N.
# Instead, we search for when Re(t) = R0 x S(t)/N < 1, i.e., S(t)/N < 1/R0

# Estimate growth rate from slope of log I during initial growth
growth_rate_mean = slope if not initial_growth_mean.empty else None
growth_rate_median = slope_m if not initial_growth_median.empty else None

# Define infectious period (assumption needed, default to 7 days)
infectious_period = 7  # days
R0_mean = 1 + growth_rate_mean*infectious_period if growth_rate_mean is not None else None
R0_median = 1 + growth_rate_median*infectious_period if growth_rate_median is not None else None

# Find time when Re(t) drops below 1, Re(t) = R0 * S(t) / N < 1

Re_below1_time_mean = None
if R0_mean is not None:
    Re_series_mean = R0_mean * mean_time_series['S'] / N_mean
    below1_indices_mean = Re_series_mean[Re_series_mean < 1].index
    if not below1_indices_mean.empty:
        Re_below1_time_mean = mean_time_series.loc[below1_indices_mean[0], 'time']

Re_below1_time_median = None
if R0_median is not None:
    Re_series_median = R0_median * median_time_series['S'] / N_median
    below1_indices_median = Re_series_median[Re_series_median < 1].index
    if not below1_indices_median.empty:
        Re_below1_time_median = median_time_series.loc[below1_indices_median[0], 'time']

# 6) Initial and final values of S
initial_S_mean = mean_time_series['S'].iloc[0]
final_S_mean = mean_time_series['S'].iloc[-1]
initial_S_median = median_time_series['S'].iloc[0]
final_S_median = median_time_series['S'].iloc[-1]

# 7) Outbreak probability (fraction of runs with nonzero final size)
# For each run, check final R > 0
final_sizes = combined_data.groupby('run').tail(1)[['run', 'R']]
outbreak_prob = (final_sizes['R'] > 0).mean()

# Return all results
results = {
    'epidemic_duration_mean_days': epidemic_duration_mean,
    'epidemic_duration_median_days': epidemic_duration_median,
    'peak_prevalence_mean': peak_mean['I'],
    'peak_prevalence_time_mean_days': peak_mean['time'],
    'peak_prevalence_median': peak_median['I'],
    'peak_prevalence_time_median_days': peak_median['time'],
    'final_epidemic_size_mean': final_R_mean,
    'final_epidemic_size_median': final_R_median,
    'doubling_time_mean_days': doubling_time_mean,
    'doubling_time_median_days': doubling_time_median,
    'R0_mean_estimated': R0_mean,
    'R0_median_estimated': R0_median,
    'effective_Re_below1_time_mean_days': Re_below1_time_mean,
    'effective_Re_below1_time_median_days': Re_below1_time_median,
    'initial_S_mean': initial_S_mean,
    'final_S_mean': final_S_mean,
    'initial_S_median': initial_S_median,
    'final_S_median': final_S_median,
    'outbreak_probability': outbreak_prob
}

results