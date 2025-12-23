
import pandas as pd, os, numpy as np
output_dir = os.path.join(os.getcwd(),'output')

er_d = pd.read_csv(os.path.join(output_dir,'results-21.csv'))
ba_d = pd.read_csv(os.path.join(output_dir,'results-22.csv'))

def metrics(df):
    I = df['I']
    E = df['E']
    N = df[['S','E','I','R']].iloc[0].sum()
    peak_I = I.max()/N
    peak_time = df['time'][I.idxmax()]
    final_size = df['R'].iloc[-1]/N
    duration = df['time'][np.where((I+E)<1)[0][0]] if any((I+E)<1) else df['time'].iloc[-1]
    return peak_I, peak_time, final_size, duration

m_er_d=metrics(er_d)
m_ba_d=metrics(ba_d)
return_dict={'metrics_er_det':m_er_d,'metrics_ba_det':m_ba_d}
