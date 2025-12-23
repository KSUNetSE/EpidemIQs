
import os

# Directory where the data resides
directory = '/Users/hosseinsamaei/phd/epidemiqs/output'

# List the files in the directory to find any other relevant CSV files
files = os.listdir(directory)

# Filter to show only CSV files
csv_files = [f for f in files if f.endswith('.csv')]