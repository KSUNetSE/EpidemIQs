
import pandas as pd, os, numpy as np
output_dir=os.path.join(os.getcwd(),'output')
res1=pd.read_csv(os.path.join(output_dir,'results-11.csv'))
res2=pd.read_csv(os.path.join(output_dir,'results-12.csv'))

def metrics(df):
    N=df['S'][0]+df['I'][0]+df['R'][0]
    peak_I=df['I'].max()
    peak_time=df['time'][df['I'].idxmax()]
    final_R=df['R'].iloc[-1]
    duration=df['time'][df['I']>0].max()
    return {'Peak_I':peak_I/N,'Peak_time':peak_time,'Final_attack_rate':final_R/N,'Duration':duration}

m1=metrics(res1)
m2=metrics(res2)
print(m1)
print(m2)
