
# Summary stats of results-01.csv
summary_01 = data[['S_mean', 'I_mean', 'R_mean']].agg(['min', 'max'])
# Summary stats of results-11.csv
summary_11 = data_11[['S_mean', 'I_mean', 'R_mean']].agg(['min', 'max'])