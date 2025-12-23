
import pandas as pd, os, json
phase_df=pd.read_csv(os.path.join(os.getcwd(),'output','results-21.csv'))
print(phase_df.head())
outcome_counts=phase_df['outcome'].value_counts()
