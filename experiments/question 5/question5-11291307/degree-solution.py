
import numpy as np
import os

# Relax: Allow up to degree=4 for remainder as well
N = 10000
n10 = int(N * 0.1125)  # 1125 at degree 10
for n3 in range(0, N-n10+1):
    for n4 in range(0, N-n10-n3+1):
        n2 = N - n10 - n3 - n4
        if n2 < 0: continue
        sum_deg = n10*10 + n3*3 + n2*2 + n4*4
        mz = sum_deg / N
        numerator = n10*(10*9) + n3*3*2 + n2*2*1 + n4*4*3
        q = numerator / (mz * N)
        if abs(mz - 3) < 0.002 and abs(q - 4) < 0.01:
            result = dict(n2=n2, n3=n3, n4=n4, mz=mz, q=q, sum_deg=sum_deg)
            break
    else:
        continue
    break
else:
    result = None
result
