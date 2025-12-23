
# Let's validate and correct the block size calculation and print it for debugging.
junior_size = 113
senior_sizes = [65]*9 + [60]  # Currently totals to 9*65 + 60 = 585 + 60 = 645
block_sizes = [junior_size] + senior_sizes
print('Block sizes:', block_sizes)
print('Total students:', sum(block_sizes))
# Check which step fails.