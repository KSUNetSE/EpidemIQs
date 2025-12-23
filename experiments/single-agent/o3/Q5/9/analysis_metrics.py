
import pandas as pd, os, numpy as np
metrics=[]
for label in [11,12]:
    path=os.path.join(os.getcwd(),'output',f'results-{label}.csv')
    df=pd.read_csv(path)
    # Assuming time sequential rows duplicates for nsim runs? fastgemf appends? Actually get_results returns last run, so metrics per final simulation only.
    # compute metrics
    final_size=df['R'].iloc[-1]/(df['S'].iloc[0]+df['I'].iloc[0]+df['R'].iloc[0])
    peak_I=df['I'].max()/ (df['S'].iloc[0]+df['I'].iloc[0]+df['R'].iloc[0])
    peak_time=df.loc[df['I'].idxmax(),'time']
    duration=df['time'].iloc[-1]
    metrics.append({'scenario':label,'final_size':final_size,'peak_I':peak_I,'peak_time':peak_time,'duration':duration})
print(metrics)
