
import pandas as pd, os
output_dir=os.path.join(os.getcwd(),'output')
for tag in ['11','12']:
    df=pd.read_csv(os.path.join(output_dir,f'results-{tag}.csv'))
    I1=df['I1']; I2=df['I2']
    final_I1=I1.iloc[-1]; final_I2=I2.iloc[-1]
    peak_I1=I1.max(); peak_I2=I2.max()
    print(tag, final_I1, final_I2, peak_I1, peak_I2)
