
# load results
import pandas as pd, os, numpy as np
results1=pd.read_csv(os.path.join(os.getcwd(),'output','results-11.csv'))
results2=pd.read_csv(os.path.join(os.getcwd(),'output','results-12.csv'))
peak_I1=results1['I'].max()/10000
peak_I2=results2['I'].max()/10000
final_size1=results1['R'].iloc[-1]/10000
final_size2=results2['R'].iloc[-1]/10000
print('random vaccination 75% peak',peak_I1,'final',final_size1)
print('k10 vaccination peak',peak_I2,'final',final_size2)