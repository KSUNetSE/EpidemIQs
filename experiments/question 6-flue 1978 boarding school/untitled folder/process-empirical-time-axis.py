
import numpy as np

# Calculate empirical time axis as days from outbreak start date
empirical_data['Date'] = pd.to_datetime(empirical_data['Date'])
start_date = empirical_data['Date'].iloc[0]
empirical_data['Days_from_start'] = (empirical_data['Date'] - start_date).dt.days

# Extract empirical B and C time series and days
empirical_days = empirical_data['Days_from_start'].values
empirical_B = empirical_data['Confined_to _bed'].values
empirical_C = empirical_data['Convalescent'].values

# Sanity check output
empirical_data[['Date','Days_from_start','Confined_to _bed','Convalescent']].head()