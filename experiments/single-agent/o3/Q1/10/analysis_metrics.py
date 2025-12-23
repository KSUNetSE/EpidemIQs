
import pandas as pd, numpy as np, os
current_dir=os.getcwd(); output_dir=os.path.join(current_dir,'output')
hm=pd.read_csv(os.path.join(output_dir,'results-11.csv'))
er=pd.read_csv(os.path.join(output_dir,'results-12.csv'))
ba=pd.read_csv(os.path.join(output_dir,'results-13.csv'))

def metrics(df):
    peak_I=int(df['I'].max())
    peak_time=int(df['time'][df['I'].idxmax()])
    final_size=int(df['R'].iloc[-1])
    duration=int(df['time'][df['I']<1].iloc[0]) if (df['I']<1).any() else int(df['time'].iloc[-1])
    return {'peak_I':peak_I,'peak_time':peak_time,'final_size':final_size,'duration':duration}

m_hom=metrics(hm); m_er=metrics(er); m_ba=metrics(ba)
all_metrics={'homogeneous':m_hom,'ER':m_er,'BA':m_ba}

import pandas as pd, os, numpy as np
output_dir=os.path.join(os.getcwd(),'output')
res_er=pd.read_csv(os.path.join(output_dir,'results-11.csv'))
res_ba=pd.read_csv(os.path.join(output_dir,'results-12.csv'))
# Metrics
# peak infected proportion and time, final size R, epidemic duration defined last time I > 1? We'll approximate.
N=5000
metrics={}
for name,df in [('ER',res_er),('BA',res_ba)]:
    peak_I=df['I'].max()/N
    peak_time=df.loc[df['I'].idxmax(),'time']
    final_R=df['R'].iloc[-1]/N
    # duration until I<1
    infected=df['I']
    idx=np.where(infected<1)[0]
    duration=df['time'].iloc[idx[0]] if len(idx)>0 else df['time'].iloc[-1]
    metrics[name]={'peak_I':peak_I,'peak_time':peak_time,'final_R':final_R,'duration':duration}
return_metrics=metrics
