
import pandas as pd, os, json
metrics = {'ER': {'peak_I': 518.0, 'peak_prop':0.1036, 'peak_time':57.79, 'final_R':3976,'attack_rate':0.7952,'duration':116.11},
          'BA': {'peak_I':162.0,'peak_prop':0.0324,'peak_time':33.27,'final_R':1365,'attack_rate':0.273,'duration':157.24}}

df = pd.DataFrame(metrics).T
print(df)