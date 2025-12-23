
# Prepare a summary table with simulation metrics for the Results and Discussion sections
import pandas as pd
metrics = {'epidemic_duration':59.34,'peak_infection':92,'peak_time':31.10,'final_epidemic_size':535,'doubling_time':float('inf')}
summary_table = pd.DataFrame([metrics])
table_path = os.path.join(os.getcwd(),'output','metrics_table.csv')
summary_table.to_csv(table_path,index=False)
return_vars=['table_path']