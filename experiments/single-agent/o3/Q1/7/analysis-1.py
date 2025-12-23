
# analysis-1.py compute metrics for three scenarios
import os, pandas as pd, numpy as np
os.makedirs(os.path.join(os.getcwd(),'output'), exist_ok=True)
scenarios={'deterministic':'results-11.csv','ER':'results-12.csv','BA':'results-13.csv'}
metrics=[]
for name,file in scenarios.items():
    df=pd.read_csv(os.path.join(os.getcwd(),'output',file))
    I=df['I'].values
    R=df['R'].values
    t=df['time'].values
    peak_I=I.max(); peak_time=t[I.argmax()]
    final_size=R[-1]
    # epidemic duration: time until I<1 afterwards; define first time after peak with I<1
    duration= t[-1]
    for tt,ii in zip(t[I.argmax():],I[I.argmax():]):
        if ii<1:
            duration=tt;break
    metrics.append({'scenario':name,'peak_I':peak_I,'peak_time':peak_time,'final_size':final_size,'duration':duration})
import pandas as pd
met_df=pd.DataFrame(metrics)
met_df.to_csv(os.path.join(os.getcwd(),'output','analysis_metrics.csv'),index=False)
