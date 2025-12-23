
import numpy as np
import os

# Coarse solution: try only multiples of 10 for n2, n3, n4, for fast search.
N = 10000
n10 = int(N * 0.1125)
found = None
for n4 in range(0, N-n10+1, 10):
    for n3 in range(0, N-n10-n4+1, 10):
        n2 = N - n10 - n3 - n4
        if n2 < 0: continue
        sum_deg = n10*10 + n4*4 + n3*3 + n2*2
        mz = sum_deg / N
        numerator = n10*90 + n4*12 + n3*6 + n2*2
        q = numerator / (mz*N)
        # look for close to target
        if abs(mz-3) < 0.005 and abs(q-4) < 0.02:
            found = dict(n2=n2, n3=n3, n4=n4, mz=mz, q=q, sum_deg=sum_deg)
            break
    if found: break
if found is None:
    found = 'NONE'
found
