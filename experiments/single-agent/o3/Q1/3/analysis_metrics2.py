
import pandas as pd, os, numpy as np
output_dir=os.path.join(os.getcwd(),'output')
df_er=pd.read_csv(os.path.join(output_dir,'results-21.csv'))
df_ba=pd.read_csv(os.path.join(output_dir,'results-22.csv'))
metrics={}
for name,df in [('ER',df_er),('BA',df_ba)]:
    peak_I = df['I'].max()
    peak_time = int(df.loc[df['I'].idxmax(),'time'])
    final_size = df.iloc[-1]['R'] / (df.iloc[-1]['S']+df.iloc[-1]['E']+df.iloc[-1]['I']+df.iloc[-1]['R'])
    duration = df['time'].max()
    metrics[name]={'peak_I':peak_I,'peak_time':peak_time,'final_size':final_size,'duration':duration}
print(metrics)