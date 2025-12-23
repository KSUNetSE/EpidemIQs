
import pandas as pd
import os

# List of all result files (all results-ij.csv files for i=0..15 and j=0,1,2 follows a pattern, but here we are given a list of 48 file paths to check first few files to understand structure
file_paths = [
    '/Users/hosseinsamaei/phd/epidemiqs/output/results-00.csv',
    '/Users/hosseinsamaei/phd/epidemiqs/output/results-01.csv',
    '/Users/hosseinsamaei/phd/epidemiqs/output/results-02.csv',
    '/Users/hosseinsamaei/phd/epidemiqs/output/results/results-03.csv',
    '/Users/hosseinsamaei/phd/epidemiqs/output/results-04.csv'
]

# Let's load the first file and check header and some rows
sample_df = pd.read_csv(file_paths[0])
columns = sample_df.columns.tolist()
sample_head = sample_df.head(10)
sample_tail = sample_df.tail(10)

columns, sample_head, sample_tail
# Check if parameters are in the first few lines or metadata in the file (like comments or text before data - sometimes files store parameters as comments before csv data)
# Using standard read_csv ignores comments, so let's open the file as text
file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-00.csv'

with open(file_path, 'r') as f:
    first_20_lines = [next(f) for _ in range(20)]

first_20_lines_text = ''.join(first_20_lines)
# Correct file paths by removing any unexpected folder name duplication and validate existence
fixed_file_paths = []
invalid_paths = []

for p in file_paths:
    # Remove duplicated 'results/' folder if accidentally inserted
    corrected_p = p.replace('/results/results-', '/results-')
    if os.path.isfile(corrected_p):
        fixed_file_paths.append(corrected_p)
    else:
        invalid_paths.append(corrected_p)

len(fixed_file_paths), invalid_paths
results_summary = []

for path in fixed_file_paths:
    df = pd.read_csv(path)
    # Parse scenario i,j from filename
    match = re.search(r'results-(\d{2})\.csv', os.path.basename(path))
    scenario = None
    if match:
        scenario_id = match.group(1)
        if len(scenario_id) == 2:
            i = int(scenario_id[0])
            j = int(scenario_id[1])
            scenario = (i,j)
        else:
            i = int(scenario_id)
            j = 0
            scenario = (i,j)
    
    # Extract prevalence columns and time
    time = df['time']
    I1 = df['I1']
    I2 = df['I2']
    
    # Calculate last 10% time threshold
    t_max = time.max()
    cutoff_time = t_max * 0.9
    late_time_mask = time >= cutoff_time
    
    total_pop = df['S'] + df['I1'] + df['I2']
    late_I1_mean = (I1[late_time_mask] / total_pop[late_time_mask]).mean()
    late_I2_mean = (I2[late_time_mask] / total_pop[late_time_mask]).mean()
    
    peak_I1_idx = I1.idxmax()
    peak_I1_val = I1.loc[peak_I1_idx] / total_pop.loc[peak_I1_idx]
    peak_I1_time = time.loc[peak_I1_idx]
    peak_I2_idx = I2.idxmax()
    peak_I2_val = I2.loc[peak_I2_idx] / total_pop.loc[peak_I2_idx]
    peak_I2_time = time.loc[peak_I2_idx]
    
    final_I1 = I1.iloc[-1] / total_pop.iloc[-1]
    final_I2 = I2.iloc[-1] / total_pop.iloc[-1]
    
    if final_I1 > 0.05 and final_I2 < 0.01:
        outcome = 'I1-dominates'
    elif final_I2 > 0.05 and final_I1 < 0.01:
        outcome = 'I2-dominates'
    elif final_I1 > 0.02 and final_I2 > 0.02:
        outcome = 'Coexist'
    elif final_I1 < 0.01 and final_I2 < 0.01:
        outcome = 'Extinct'
    else:
        outcome = 'Unclassified'
    
    results_summary.append({
        'scenario': scenario,
        'late_I1_mean': late_I1_mean,
        'late_I2_mean': late_I2_mean,
        'peak_I1_val': peak_I1_val,
        'peak_I1_time': peak_I1_time,
        'peak_I2_val': peak_I2_val,
        'peak_I2_time': peak_I2_time,
        'final_I1': final_I1,
        'final_I2': final_I2,
        'outcome': outcome
    })

results_summary