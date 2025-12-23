
import pandas as pd, os
out=os.path.join(os.getcwd(),'output')
res11 = pd.read_csv(os.path.join(out,'results-11.csv'))
res12 = pd.read_csv(os.path.join(out,'results-12.csv'))
final11 = res11.tail(1)
final12 = res12.tail(1)
print('random75 final', final11[['S','I','R']].values)
print('deg10 vacc final', final12[['S','I','R']].values)