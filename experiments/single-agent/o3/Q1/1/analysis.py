
import pandas as pd, os
output_dir=os.path.join(os.getcwd(),'output')
res_er=pd.read_csv(os.path.join(output_dir,'results-11.csv'))
res_ba=pd.read_csv(os.path.join(output_dir,'results-12.csv'))
metrics={} 
for name,res in [('ER',res_er),('BA',res_ba)]:
    total=res[['E','I']].max(axis=1).sum()
