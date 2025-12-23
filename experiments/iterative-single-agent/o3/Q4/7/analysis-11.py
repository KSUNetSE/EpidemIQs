
import pandas as pd, os, numpy as np, matplotlib.pyplot as plt
output_dir=os.path.join(os.getcwd(),'output')
df=pd.read_csv(os.path.join(output_dir,'results-11.csv'))
I1=df['I1']; I2=df['I2']
final_I1=I1.iloc[-1]; final_I2=I2.iloc[-1]
peak_I1=I1.max(); peak_I2=I2.max()
print(final_I1, final_I2, peak_I1, peak_I2)

import pandas as pd, os, numpy as np
fp=os.path.join(os.getcwd(),'output','results-11.csv')
print('file exist', os.path.exists(fp))
df=pd.read_csv(fp)
I1_final=df['I1'].iloc[-1]
I2_final=df['I2'].iloc[-1]
print(I1_final,I2_final)
