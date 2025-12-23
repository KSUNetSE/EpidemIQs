
import pandas as pd, os, numpy as np
out=os.path.join(os.getcwd(),'output')
metrics={}
for tag in ['11','12']:
    df=pd.read_csv(os.path.join(out,f'results-{tag}.csv'))
    I1=df['I1']; I2=df['I2']
    metrics[tag]={'final_I1':int(I1.iloc[-1]),'final_I2':int(I2.iloc[-1]),'peak_I1':int(I1.max()),'peak_I2':int(I2.max())}
print(metrics)
