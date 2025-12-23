
# We have additional csv files like 'results-11.csv', 'results-12.csv', etc. that may contain time series for B(t) or prevalence
# Check the structure of one of those result files, e.g. results-11.csv
results_11 = pd.read_csv('output/results-11.csv')
results_11.head()