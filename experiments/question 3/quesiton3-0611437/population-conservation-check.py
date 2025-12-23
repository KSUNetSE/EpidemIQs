
# Check population conservation for each dataset, total population should be N=1000
N = 1000
pop_check_11 = (data_11['S'] + data_11['I'] + data_11['R']).round(0) == N
pop_check_12 = (data_12['S'] + data_12['I'] + data_12['R']).round(0) == N
pop_consistent_11 = pop_check_11.all()
pop_consistent_12 = pop_check_12.all()
pop_consistent_11, pop_consistent_12