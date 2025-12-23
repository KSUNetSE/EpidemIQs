
import pandas as pd

# Load the data to examine its contents
file_path = '/Users/hosseinsamaei/phd/gemf_llm/output/results-51.csv'
data = pd.read_csv(file_path)
data.head(), data.columns.tolist()
import pandas as pd

# Load the data to inspect its structure
file_path = '/Users/hosseinsamaei/phd/gemf_llm/output/results-61.csv'
data = pd.read_csv(file_path)

# Display the first few rows and columns info to understand its structure
head = data.head()
info = data.info()

head, info
# Brief analysis of the dataset structure and runs
# Data columns: time, S (susceptible count), I (infected count), R (recovered count)
# It appears to be the output of an epidemic simulation over time

# Let's look at general statistics and unique time points
n_samples = len(data)
unique_times = data['time'].nunique()

# Look for runs info - check if 'run' or similar column exists
has_run_col = 'run' in data.columns

n_samples, unique_times, has_run_col
# The data has columns ['time', 'S', 'I', 'R'], which presumably correspond to time, Susceptible, Infectious, and Recovered individuals respectively.
# Since there are 100 runs, check how results for multiple runs are stored: see if there's a 'run' or similar column or if data is stacked.

runs_info = data['time'].nunique()  # number of unique time steps

# Inspect if there's any run identifier by checking columns
cols = data.columns.tolist()

cols, runs_info
# There is no explicit run identifier column, and the total unique time points are 2885,
# which likely means the data from all 100 runs are combined or averaged already, or runs are concatenated without run IDs.
# Let's check basic statistics for S, I, R to understand if these are raw counts or averaged values.

summary_stats = data[['S', 'I', 'R']].describe()

# Check if all data is summed or averaged at each time
min_time = data['time'].min()
max_time = data['time'].max()

summary_stats, min_time, max_time
# There is no run column, so data might be for a single run or all runs aggregated
# To clarify if multiple runs are stacked vertically and time is reset or not
# Let's check the time distribution and see if there are repetitions in time

time_counts = data['time'].value_counts()

# Check how many unique times correspond to the minimum and maximum times
min_time = data['time'].min()
max_time = data['time'].max()

# Display some rows near the start and end
start_rows = data.head(10)
end_rows = data.tail(10)

time_counts.head(), time_counts.tail(), min_time, max_time, start_rows, end_rows
# Since data columns are counts, we'll convert them to fractions by dividing by population N=10000 to compute metrics in fraction units.
population = 10000

data['S_frac'] = data['S'] / population
data['I_frac'] = data['I'] / population
data['R_frac'] = data['R'] / population

# Since we expect 100 runs and the data doesn't have explicit run IDs, let's check if the data is grouped or averaged already.
# Let's check the distribution of counts of repeated time points to find if multiple runs are stacked by time.
time_counts = data['time'].value_counts()
time_counts_sorted = time_counts.sort_index()
time_counts_sorted.head(), time_counts_sorted.unique()  # check how many records per time point

# From the data it looks like this is a single run of the model with 10000 total population (S+I+R around 10000)
# No 'run' column means no explicit multiple runs stored here, just one run's full time series

# Let's calculate directly the requested metrics for this run and then see if further files or data about other runs exist

N = 10000

# Calculate final epidemic size (fraction ever infected = R/N at the end)
final_epidemic_size = data['R'].iloc[-1] / N

# Peak infection rate (max I/N and corresponding time)
peak_infection_index = data['I'].idxmax()
peak_infection_fraction = data.loc[peak_infection_index, 'I'] / N
peak_infection_time = data.loc[peak_infection_index, 'time']

# Epidemic duration: from first nonzero infected to time when infected returns near zero
first_nonzero_infected_time = data[data['I']>0]['time'].iloc[0]
# Define near zero as I <= 1 individual
near_zero_time = data[data['I']<=1]['time'].max()
epidemic_duration = near_zero_time - first_nonzero_infected_time

# Outbreak velocity: time to reach half of final epidemic size R(t) = 0.5 * final_epidemic_size * N
half_final_R = 0.5 * data['R'].iloc[-1]
half_final_time = data[data['R'] >= half_final_R]['time'].iloc[0]

# Results with units
results = {
    'final_epidemic_size_fraction': final_epidemic_size,  # fraction of population (unitless)
    'peak_infection_fraction': peak_infection_fraction,    # fraction of population (unitless)
    'peak_infection_time': peak_infection_time,            # time units (assumed days or simulation time units)
    'epidemic_duration': epidemic_duration,                # time units
    'outbreak_velocity_time': half_final_time,             # time units
}
results
# Since we originally expected 100 runs but dataset has only one run, let's see if other runs are stored separately or in other files
# For completeness, list files in output directory to check for similar result files
import os
output_dir = '/Users/hosseinsamaei/phd/gemf_llm/output/'
all_files = os.listdir(output_dir)
all_files
# Fix the syntax error (removed wrong line break characters) and retry

files_summary = {}
for file in files_to_check:
    try:
        df = pd.read_csv(os.path.join(output_dir, file))
        files_summary[file] = df.columns.tolist()
    except Exception as e:
        files_summary[file] = str(e)
files_summary
# The other results files appear to contain error messages instead of valid data
# Only results-51.csv and results-61.csv contain the valid columns time, S, I, R
# Let's load results-61.csv and extract metrics similarly

results_61 = pd.read_csv(os.path.join(output_dir, 'results-61.csv'))

# Check its basic info and head
results_61.head(), results_61.columns.tolist(), results_61.shape
# Same metrics extraction for results-61.csv

final_epidemic_size_61 = results_61['R'].iloc[-1] / N
peak_infection_index_61 = results_61['I'].idxmax()
peak_infection_fraction_61 = results_61.loc[peak_infection_index_61, 'I'] / N
peak_infection_time_61 = results_61.loc[peak_infection_index_61, 'time']
first_nonzero_infected_time_61 = results_61[results_61['I']>0]['time'].iloc[0]
near_zero_time_61 = results_61[results_61['I']<=1]['time'].max()
epidemic_duration_61 = near_zero_time_61 - first_nonzero_infected_time_61
half_final_R_61 = 0.5 * results_61['R'].iloc[-1]
half_final_time_61 = results_61[results_61['R'] >= half_final_R_61]['time'].iloc[0]

results_61_metrics = {
    'final_epidemic_size_fraction': final_epidemic_size_61,
    'peak_infection_fraction': peak_infection_fraction_61,
    'peak_infection_time': peak_infection_time_61,
    'epidemic_duration': epidemic_duration_61,
    'outbreak_velocity_time': half_final_time_61,
}
results_61_metrics
# Review the statistics of the data to see if there is variability to calculate std across runs
# Since there is no run ID, check std values for final R, max I, and other points to infer variability

# Max R and corresponding std at last time point
last_time = data['time'].max()
data_at_last_time = data.loc[data['time'] == last_time]

# To check if multiple runs data is in this file, verify if there are multiple entries at same time point (previous check showed only 1)
# Instead check std over window of last few points as proxy

data['time_rounded'] = data['time'].round(3)
time_groups = data.groupby('time_rounded')
final_R_values = time_groups['R_frac'].last()  # take last as proxy for final R from groups

# Since only 1 per time, no multiple runs to get std - so presume single run or averaged data
# So, we cannot calculate std across runs without multiple run data

final_epidemic_size_std = None
peak_infectious_fraction_std = None
peak_infectious_time_std = None
epidemic_duration_std = None
reach_half_R_std = None

# Prepare results summary in decimal fractions with units converted to fractions and times in model time units (likely days or arbitrary units)

metrics = {
    "Final epidemic size (fraction infected)": (final_epidemic_size, final_epidemic_size_std, "fraction of population, unitless"),
    "Peak infection rate (max fraction infected)": (peak_infectious_fraction, peak_infectious_fraction_std, "fraction of population, unitless"),
    "Time of peak infection": (peak_infectious_time, peak_infectious_time_std, "time units"),
    "Epidemic duration": (epidemic_duration, epidemic_duration_std, "time units"),
    "Outbreak velocity (time to half final size)": (reach_half_R, reach_half_R_std, "time units"),
}
metrics
# The metrics for results-51.csv and results-61.csv are identical, possibly they are from the same run/configuration
# Since there is no run information and no multiple runs in these files, uncertainty stats cannot be computed from these

# Provide a summary of the metrics with units and calculation details
summary = {
    'Population size': N,  # individuals
    'Final epidemic size': {
        'value': results['final_epidemic_size_fraction'],
        'units': 'fraction of total population',
        'calculation': 'R(t) at final simulation time divided by N'
    },
    'Peak infection rate': {
        'value': results['peak_infection_fraction'],
        'time_of_peak': results['peak_infection_time'],
        'units': 'fraction of total population and simulation time units',
        'calculation': 'Max I(t)/N and corresponding time'
    },
    'Epidemic duration': {
        'value': results['epidemic_duration'],
        'units': 'simulation time units',
        'calculation': 'Time from first nonzero I to near zero (<=1) I'
    },
    'Outbreak velocity': {
        'value': results['outbreak_velocity_time'],
        'units': 'simulation time units',
        'calculation': 'Time when R(t) reaches half of final epidemic size'
    },
    'Notes': 'No multiple runs present in data, hence no std or variance metrics can be reported.'
}
summary