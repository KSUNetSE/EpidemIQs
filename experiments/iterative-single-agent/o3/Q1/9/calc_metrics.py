
import pandas as pd, numpy as np, os, math
output_dir = os.path.join(os.getcwd(),'output')
er = pd.read_csv(os.path.join(output_dir,'results-11.csv'))
ba = pd.read_csv(os.path.join(output_dir,'results-12.csv'))

def metrics(df):
    I = df['I']
    E = df['E']
    N = df[['S','E','I','R']].iloc[0].sum()
    peak_I = I.max()/N
    peak_time = df['time'][I.idxmax()]
    final_size = df['R'].iloc[-1]/N
    duration = df['time'][np.where((I+E)<1)[0][0]] if any((I+E)<1) else df['time'].iloc[-1]
    return peak_I, peak_time, final_size, duration

metrics_er = metrics(er)
metrics_ba = metrics(ba)
return_dict={'metrics_er':metrics_er,'metrics_ba':metrics_ba}
