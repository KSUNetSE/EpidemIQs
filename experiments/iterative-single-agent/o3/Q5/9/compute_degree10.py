
import math, numpy as np
z=3
p=[math.exp(-z)*z**k/math.factorial(k) for k in range(0,21)]
p=[x/sum(p) for x in p]
mean=sum(k*p[k] for k in range(len(p)))
second=sum(k*k*p[k] for k in range(len(p)))
q=(second-mean)/mean
print('mean',mean,'second',second,'q',q)
print('p10=',p[10])
print('vacc remove degree 10 (all)')
mean2=sum(k*p[k] for k in range(len(p)) if k!=10)
second2=sum(k*k*p[k] for k in range(len(p)) if k!=10)
q2=(second2-mean2)/mean2
print('mean2',mean2,'q2',q2)
frac_removed=p[10]
print('frac removed',frac_removed)