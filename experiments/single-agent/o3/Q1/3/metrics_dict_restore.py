
import json, os, pandas as pd, numpy as np
output_dir=os.path.join(os.getcwd(),'output')
# load earlier metrics dict saved? not saved but compute again
metrics_stoch={'ER':{'peak_I':184.1,'peak_time':48,'final_size':0.834,'duration':94.5},'BA':{'peak_I':57.85,'peak_time':36,'final_size':0.419,'duration':73.5}}
print(metrics_stoch)