
import pandas as pd, os, numpy as np, glob, json, math
output_dir = os.path.join(os.getcwd(),'output')
labels = ['baseline','random75','targeted_deg10_all','targeted_deg10_8']
metrics = {}
for idx,lab in enumerate(labels):
    df = pd.read_csv(os.path.join(output_dir, f'results-1{idx}.csv'))
    final_R = df['R'].iloc[-1]
    peak_I = df['I'].max()
    peak_time = df['time'][df['I'].idxmax()]
    metrics[lab] = {'final_R':int(final_R), 'peak_I':int(peak_I), 'peak_time':float(peak_time)}

metrics