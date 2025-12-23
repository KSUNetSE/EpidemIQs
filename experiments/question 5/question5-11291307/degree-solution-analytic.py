
import numpy as np
import os

# Next: Use a solver for ax=b to get closest solution for n2 and n3 after fixing n10, allow remainder to be fixed (~nearest integer to minimize error)
N = 10000
n10 = int(N * 0.1125)  # 1125
k10 = 10
#
# The system:
# n2 + n3 = N - n10
# 2*n2 + 3*n3 = S, with S = Nz - 10*n10 = 30000 - 11250 = 18750
# 2*n2 + 6*n3 = E, with E = Nqz - 90*n10 = 120000 - 101250 = 18750
#
# From above, find n3, n2. This only possible if exactly
# (2*n2 + 6*n3) - (2*n2 + 3*n3) = 3*n3 = E-S = 0 => n3=0
# So, n2 = N-n10 = 8875
# Check if possible
sum_deg = n10*10 + 8875*2
mz = sum_deg/N
numerator = n10*10*9 + 8875*2*1
q = numerator / (mz*N)
# Print solution
sol = dict(n10=n10, n3=0, n2=8875, mz=mz, q=q, sum_deg=sum_deg)
sol
