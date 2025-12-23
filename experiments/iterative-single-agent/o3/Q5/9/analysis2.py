
import pandas as pd, os, numpy as np
metrics=[]
for label in [11,12]:
    df=pd.read_csv(os.path.join(os.getcwd(),'output',f'results-{label}.csv'))
    Ntotal=df[['S','I','R']].iloc[0].sum()
    R0=df['R'].iloc[0]
    final_R=df['R'].iloc[-1]
    infected_added = final_R - R0
    final_size=infected_added/Ntotal
    peak_I=df['I'].max()/Ntotal
    peak_time=df.loc[df['I'].idxmax(),'time']
    duration=df[df['I']>0]['time'].max()
    metrics.append({'scenario':label,'vaccinated':R0/Ntotal,'final_attack_rate':final_size,'peak_I':peak_I,'peak_time':peak_time,'duration':duration})
print(metrics)
