
import pandas as pd, os, numpy as np
res1=pd.read_csv(os.path.join(os.getcwd(),'output','results-11.csv'))
res2=pd.read_csv(os.path.join(os.getcwd(),'output','results-12.csv'))
peak_I1=res1['I'].max()/10000
peak_I2=res2['I'].max()/10000
final_size1=res1['R'].iloc[-1]/10000
final_size2=res2['R'].iloc[-1]/10000
print(peak_I1, peak_I2, final_size1, final_size2)