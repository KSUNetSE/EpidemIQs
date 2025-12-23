
# Read the size of each file to see scale of data
file_sizes = {}
for f in ['/Users/hosseinsamaei/phd/epidemiqs/output/results-00.csv', '/Users/hosseinsamaei/phd/epidemiqs/output/results-03.csv']:
    df = pd.read_csv(f)
    file_sizes[f] = df.shape
file_sizes