
# Double-check if vaccination info (number/fraction of nodes vaccinated) is present in any single-run file columns
# We'll check the first file that contains S, I, R columns and stats to see if any other columns exist that denote vaccination

df_check = pd.read_csv('output/results-11.csv')
df_check.columns