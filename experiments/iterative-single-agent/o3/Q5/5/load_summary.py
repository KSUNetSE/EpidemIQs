
import pandas as pd, os
summary=pd.read_csv(os.path.join(os.getcwd(),'output','summary_results.csv'))
print(summary)
