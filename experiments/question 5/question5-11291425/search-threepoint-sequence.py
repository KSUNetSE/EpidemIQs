
# Let's try building a 3-point discrete distribution: p_a at degree a, p_b at degree b, p_10 at degree 10.
# We'll try a=2, b=3.
# Want: p_2*a + p_3*b + p_10*10 = 3,
#       p_2*a^2 + p_3*b^2 + p_10*100 = 15,
#       p_2 + p_3 + p_10 = 1.
import numpy as np

A = np.array([
    [2, 3, 10],
    [2**2, 3**2, 100],
    [1,   1,    1]
])
b = np.array([3, 15, 1])
sol = np.linalg.solve(A, b)
p2, p3, p10 = sol
# ensure all in [0,1] and sum closest to 1
p2, p3, p10 = [max(0,x) for x in [p2,p3,p10]]
s = p2+p3+p10
p2/=s; p3/=s; p10/=s
N=10000
n2 = int(np.round(p2*N))
n3 = int(np.round(p3*N))
n10 = N-n2-n3
sequence = [2]*n2 + [3]*n3 + [10]*n10
np.random.shuffle(sequence)

# Guarantee even sum
if sum(sequence)%2 != 0:
    for i in range(len(sequence)):
        if sequence[i]>0:
            sequence[i]-=1
            break
mean_k = np.mean(sequence)
second_mom = np.mean(np.array(sequence)**2)
q = (second_mom - mean_k)/mean_k
frac_10 = n10/N
return_vars = ['mean_k','second_mom','q','frac_10','n2','n3','n10','sequence']