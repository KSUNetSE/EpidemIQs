
import os

# Directory containing current data
data_dir = '/Users/hosseinsamaei/phd/epidemiqs/output/'

# Look for other files in the directory related to ER scenario
all_files = os.listdir(data_dir)
er_files = [f for f in all_files if 'ER' in f or 'er' in f or 'Erdos' in f]
er_files