
import pandas as pd, os
out_dir=os.path.join(os.getcwd(),'output')
results_random=pd.read_csv(os.path.join(out_dir,'results-11.csv')) if os.path.exists(os.path.join(out_dir,'results-11.csv')) else None
# We already produced network. Now plan for simulation code to be written later