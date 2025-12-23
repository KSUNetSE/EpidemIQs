
import pandas as pd, os, numpy as np
out=os.path.join(os.getcwd(),'output')
res_er=pd.read_csv(os.path.join(out,'results-11.csv'))
res_ba=pd.read_csv(os.path.join(out,'results-12.csv'))

def metrics(res):
    peak_I=res['I'].max()
    time_peak=res.loc[res['I'].idxmax(),'time']
    final_R=res['R'].iloc[-1]
    epidemic_duration=res[res['I']>1].shape[0] if (res['I']>1).any() else 0
    return peak_I,time_peak,final_R,epidemic_duration

m_er=metrics(res_er)
m_ba=metrics(res_ba)
