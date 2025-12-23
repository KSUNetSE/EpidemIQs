
# As the vision agent failed, provide a narrative summary by further extracting S, E, I, R curves at key points
import pandas as pd
import os

summary = {}
for i, j, name in [(1,1,'er'), (1,2,'ba')]:
    p = os.path.join(os.getcwd(),'output',f'results-{i}{j}.csv')
    df = pd.read_csv(p)
    initial = {comp: float(df.loc[0,comp]) for comp in ['S','E','I','R']}
    at_peak = {comp: float(df.loc[df['I'].idxmax(),comp]) for comp in ['S','E','I','R']}
    at_end  = {comp: float(df.iloc[-1][comp]) for comp in ['S','E','I','R']}
    summary[name] = dict(initial=initial,at_peak=at_peak,at_end=at_end)
summary