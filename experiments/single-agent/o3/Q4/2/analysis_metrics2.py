
import pandas as pd, os, numpy as np

df2=pd.read_csv(os.path.join(os.getcwd(),'output','results-12.csv'))
metrics2={'final_I1':int(df2['I1'].iloc[-1]), 'final_I2':int(df2['I2'].iloc[-1]), 'peak_I1':int(df2['I1'].max()), 'peak_I2':int(df2['I2'].max()), 'peak_time_I1': float(df2.loc[df2['I1'].idxmax(),'time']), 'peak_time_I2': float(df2.loc[df2['I2'].idxmax(),'time'])}
