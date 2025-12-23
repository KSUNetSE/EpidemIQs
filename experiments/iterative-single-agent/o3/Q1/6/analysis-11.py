
import pandas as pd, os, numpy as np, matplotlib.pyplot as plt
out_dir=os.path.join(os.getcwd(),'output')
paths={'er':'results-111.csv','ba':'results-112.csv'}
metrics={}
for tag,fname in paths.items():
    df=pd.read_csv(os.path.join(out_dir,fname))
    I=df['I']
    peak=I.max(); peak_time=df.loc[I.idxmax(),'time']
    final_size=df.iloc[-1]['R']
    metrics[tag]={'peak_I':peak,'peak_time':peak_time,'final_R':final_size}
print(metrics)
