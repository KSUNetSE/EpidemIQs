
# Prepare a latex table for the metrics
import pandas as pd
import os
df = pd.read_csv(os.path.join(os.getcwd(), 'output', 'summary_metrics.csv'))
latex = df.to_latex(index=False, float_format='%.2f')
with open(os.path.join(os.getcwd(), 'output', 'table_metrics.tex'), 'w') as f:
    f.write(latex)
latex