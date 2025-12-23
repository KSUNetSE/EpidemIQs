
# Since the peak prevalence, peak time, duration, and CI widths are not directly in these metric files,
# we need to try to find those in other related files for each scenario.
# Let's check if there are other files that contain time series or parameter info to extract those values.
# List files in 'output' directory
import os
files_in_output = os.listdir('output')
files_in_output