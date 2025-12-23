
# Additional inspection to check if other required values are present in metrics CSVs
# Check if there are columns or rows containing peak prevalence, peak time, epidemic duration in each metric file
peak_check = {}
for file, df in data_dict.items():
    peak_values = df[df['Unnamed: 0'].str.contains('peak', case=False, na=False)]['0'].to_dict()
    duration_values = df[df['Unnamed: 0'].str.contains('duration', case=False, na=False)]['0'].to_dict()
    ci_width_ar = df[df['Unnamed: 0'].str.contains('CI_ar_width', case=False, na=False)]['0'].to_dict()
    ci_width_b = df[df['Unnamed: 0'].str.contains('CI_B_width', case=False, na=False)]['0'].to_dict()
    peak_check[file] = {'peak_B': peak_values, 'duration': duration_values, 'ci_width_ar': ci_width_ar, 'ci_width_b': ci_width_b}
peak_check