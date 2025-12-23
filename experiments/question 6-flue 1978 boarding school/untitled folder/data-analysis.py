
import pandas as pd

# Load the data from the specified CSV file
file_path = 'output/results-12.csv'
data = pd.read_csv(file_path)

# Inspect the first few rows and columns of the data to understand its structure
initial_inspection = data.head()
data_info = data.info()

# Get the column names to check relevant columns
column_names = data.columns.tolist()

initial_inspection, column_names
# Extract relevant columns
B = data['B']
C = data['C']
time = data['time']

# Total population N
N = 763

# Peak value and timing for B
i_peak_index = B.idxmax()
B_peak_value = B.iloc[i_peak_index]
B_peak_time = time.iloc[i_peak_index]

# Final cumulative Attack Rate (AR) calculation
# We consider fraction ever bed-confined or recovered by end of simulation
# In this data, count of bed-confined (B) and convalescent (C) at the last time point
B_final = B.iloc[-1]
C_final = C.iloc[-1]

# AR = fraction of population that are or were bed-confined or convalescent
# Since data is a time series with current counts, cumulative probably can be approximated as final B + final C + final R
# Consider R (recovered) as well as convalescents are recovered but I'll just do B + C to be consistent with question
# Given typical modeling, attack rate is total fraction R+ C + B (ever infected/recovered or bed-confined)

R_final = data['R'].iloc[-1]
AR = (B_final + C_final + R_final) / N

# Find epidemic duration as days between first and last day when B or C > 1 individual
above_one = (B > 1) | (C > 1)
first_day = time[above_one].iloc[0] if above_one.any() else None
last_day = time[above_one].iloc[-1] if above_one.any() else None

if first_day is not None and last_day is not None:
    epidemic_duration = last_day - first_day
else:
    epidemic_duration = 0

# Results dictionary for clear reporting
results = {
    'B_peak_value': B_peak_value, 'B_peak_time': B_peak_time,
    'Attack_Rate_fraction': AR,
    'epidemic_duration_days': epidemic_duration,
    'time_series': {'time': time.tolist(), 'B': B.tolist(), 'C': C.tolist()}
}