
# Further analysis: load and format summary for reporting
import pandas as pd
import os

sumtab = pd.read_csv(os.path.join(os.getcwd(), 'output', 'summary-metrics.csv'))
sumtab_latex = sumtab.to_latex(index=False, float_format='%.2f')

return_vars = ['sumtab_latex']
