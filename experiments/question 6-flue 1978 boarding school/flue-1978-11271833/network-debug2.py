
# Check and fix the block size problem. There are 763 students: 1 junior house, 10 senior houses.
# The senior houses should sum to 650 (763 - 113 = 650). Even split: 65 per house for 10 houses.
junior_size = 113
num_senior = 10
senior_size = 65
block_sizes = [junior_size] + [senior_size]*num_senior
print('Block sizes:', block_sizes)
print('Total:', sum(block_sizes))
