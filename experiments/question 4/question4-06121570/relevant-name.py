
import pandas as pd

# Load the data to examine its structure
file_path = 'output/results-61.csv'
data = pd.read_csv(file_path)
data.head(), data.columns.tolist()
import pandas as pd

# Load data from the file to examine its structure
file_path = 'output/results-41.csv'
data = pd.read_csv(file_path)
data.head(), data.columns, data.shape
import pandas as pd

# Load the data from the provided file path
file_path = 'output/results-51.csv'
data = pd.read_csv(file_path)

# Display the first few rows and columns to understand the structure and columns of the data
data.head(), data.columns.tolist()
# Fix indentation and sort data

data = data.sort_values(by='time')
data.tail()  # verify sorted data
N = 1000
threshold = 0.01 * N  # 10

# 1. Final prevalence: average of last 20 time steps
final_20_I1 = data['I1'].tail(20).mean()
final_20_I2 = data['I2'].tail(20).mean()

# 2. Peak prevalence: max value and time of max
peak_I1 = data['I1'].max()
peak_time_I1 = data.loc[data['I1'].idxmax(), 'time']

peak_I2 = data['I2'].max()
peak_time_I2 = data.loc[data['I2'].idxmax(), 'time']

# 3. Time to extinction method
# Check the first time the prevalence drops below threshold after being above

def time_to_extinction(series):
    above = series > threshold
    extinct_time = np.nan
    previously_above = False
    for i in range(len(series)):
        if above.iat[i]:
            previously_above = True
        if previously_above and not above.iat[i]:
            extinct_time = data.at[i, 'time']
            break
    return extinct_time

# Calculate extinction time
extinction_I1 = time_to_extinction(data['I1'])
extinction_I2 = time_to_extinction(data['I2'])

# 4. Dominance indicator: 1 if prevalence above threshold at end
indicator_I1 = int(data['I1'].iloc[-1] > threshold)
indicator_I2 = int(data['I2'].iloc[-1] > threshold)

metrics_part1 = {
    'I1': {
        'final_prevalence': final_20_I1,
        'peak_prevalence': peak_I1,
        'peak_time': peak_time_I1,
        'time_to_extinction': extinction_I1,
        'dominance_indicator': indicator_I1,
    },
    'I2': {
        'final_prevalence': final_20_I2,
        'peak_prevalence': peak_I2,
        'peak_time': peak_time_I2,
        'time_to_extinction': extinction_I2,
        'dominance_indicator': indicator_I2,
    }
}
metrics_part1
# Find periods where both I1 and I2 are above threshold
coexist = (data['I1'] > threshold) & (data['I2'] > threshold)

# Use time points to calculate total coexistence duration
coexist_times = data['time'][coexist]

# If coexistence periods are continuous or mostly continuous, difference between last and first

if coexist_times.empty:
    coexistence_duration = 0.0
else:
    coexistence_duration = coexist_times.iloc[-1] - coexist_times.iloc[0]

metrics_part1['coexistence_duration'] = coexistence_duration
metrics_part1