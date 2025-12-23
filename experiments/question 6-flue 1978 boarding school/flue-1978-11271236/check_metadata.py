
import os

# Check if metadata or model parameters exist in any csv comments or nearby files
# Attempt to read the first few lines raw text for a file to see if parameters are stored there
path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-11.csv'

with open(path, 'r') as file:
    head_text = ''.join([next(file) for _ in range(20)])

head_text