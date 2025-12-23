
import math
lam=3.0
p = [math.exp(-lam)*lam**k/math.factorial(k) for k in range(0,30)]
P10 = p[10]
print('P10',P10)
orig_mean = sum(k*p[k] for k in range(len(p)))
orig_second = sum(k*k*p[k] for k in range(len(p)))
q = (orig_second - orig_mean)/orig_mean
print('orig q',q)
remaining_prob = 1-P10
mean_new = sum(k*p[k] for k in range(len(p)) if k!=10)/remaining_prob
second_new = sum(k*k*p[k] for k in range(len(p)) if k!=10)/remaining_prob
q_new = (second_new - mean_new)/mean_new
print('new mean',mean_new,'new q',q_new)
