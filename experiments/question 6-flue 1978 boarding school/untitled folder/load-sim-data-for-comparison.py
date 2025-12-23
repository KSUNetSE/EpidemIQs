
import pandas as pd

# Load simulation data from previous file
sim_file_path = 'output/results-12.csv'
sim_data = pd.read_csv(sim_file_path)

# Extract needed columns
sim_time = sim_data['time']
sim_B = sim_data['B']
sim_C = sim_data['C']

# Return shapes and first rows for verification
sim_data.shape, sim_data.columns, sim_data.head()