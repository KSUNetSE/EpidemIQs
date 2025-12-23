
# Correct file paths as per user provided
file_paths = [
    'output/results-21.csv',
    'output/results-22.csv',
    'output/results-23.csv',
    'output/results-24.csv',
    'output/results-25.csv',
    'output/results-26.csv'
]

all_metrics_correct = {}
for fp in file_paths:
    df = pd.read_csv(fp)
    metrics = extract_metrics(df)
    all_metrics_correct[fp] = metrics

# Extract potential beta values from file name suffix (assume last two digits are approximations)
def extract_beta_from_filename(filename):
    import re
    match = re.search(r'results-(\d+).csv', filename)
    if match:
        val = match.group(1)
        try:
            beta_val = float(val) / 10
            return beta_val
        except:
            return None
    return None

params = {}
for fp in file_paths:
    beta_val = extract_beta_from_filename(fp)
    if beta_val is None:
        beta_val = 'Unknown'
    # assign beta1=beta2=beta_val for now as guess, since delta1=delta2=1 and tau=beta
    params[fp] = {'beta1': beta_val, 'beta2': beta_val, 'delta1': 1, 'delta2': 1, 'tau1': beta_val, 'tau2': beta_val}

all_metrics_correct, params