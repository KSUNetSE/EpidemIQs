
import pandas as pd, os, numpy as np
out=os.path.join(os.getcwd(),'output')
res11 = pd.read_csv(os.path.join(out,'results-11.csv'))
res12 = pd.read_csv(os.path.join(out,'results-12.csv'))
peak_I11 = res11['I'].max()
peak_I12 = res12['I'].max()
final_R11 = res11['R'].iloc[-1]
final_R12 = res12['R'].iloc[-1]
result={'peak_I_random75': int(peak_I11), 'final_R_random75': int(final_R11),
        'peak_I_deg10': int(peak_I12), 'final_R_deg10': int(final_R12)}
