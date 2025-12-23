
import pandas as pd

# Load the data for initial examination
file_path = 'output/results-12.csv'
data = pd.read_csv(file_path)

# Basic info on the data to understand its structure
info = data.info()

# Peek the first few rows
head = data.head()

info, head, list(data.columns)
# Check if there is any run identifier column
run_id_columns = [col for col in data.columns if 'run' in col.lower() or 'id' in col.lower()]

# Check the tail of the dataset to see time coverage and final stats
tail = data.tail(10)

run_id_columns, tail
# Re-read the csv data with full rows preview to assert if data lines have different kind of formatting or indicators
import csv

with open(file_path, 'r') as file:
    lines = [next(file) for _ in range(20)]

sample_lines = lines
sample_lines