
import pandas as pd, os
csv1=os.path.join(os.getcwd(),'output','results-11.csv')
csv2=os.path.join(os.getcwd(),'output','results-12.csv')
df1=pd.read_csv(csv1)
df2=pd.read_csv(csv2)
metrics={}
for name,df in [('low',df1),('high',df2)]:
    peak_I=df['I'].max(); peak_time=df.loc[df['I'].idxmax(),'time']
    final_removed=df['R'].iloc[-1]
    S_end=df['S'].iloc[-1]
    metrics[name]={'peak_I':int(peak_I), 'peak_time':peak_time, 'final_removed':int(final_removed), 'susceptible_end':int(S_end)}
print(metrics)
