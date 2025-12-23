
import math

lambda_val = 3
k = 10
P_k10 = math.exp(-lambda_val) * (lambda_val**k) / math.factorial(k)

result = {'P_k10': P_k10}
result
