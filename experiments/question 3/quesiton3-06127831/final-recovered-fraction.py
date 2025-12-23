
# Extract the last row from the data
data_final = data.iloc[-1]

# Calculate the total population N (assuming it's constant over time)
N = data_final['S'] + data_final['I'] + data_final['R']

# Calculate the final recovered fraction
final_recovered_fraction = data_final['R'] / N

data_final, N, final_recovered_fraction
# Extract final row values
final_row = data.iloc[-1]

# Calculate total population N (assumed constant)
N = final_row['S'] + final_row['I'] + final_row['R']

# Calculate final recovered fraction
final_recovered_fraction = final_row['R'] / N

final_row, N, final_recovered_fraction