
import math
# Poisson mean 3 distribution
lam=3.0
P10=math.exp(-lam)*lam**10/math.factorial(10)
print('P10',P10)
# compute sums up to maybe 20
p=[]
for k in range(0,30):
    p_k=math.exp(-lam)*lam**k/math.factorial(k)
    p.append(p_k)

sum_all=sum(p)
# confirm ~1
mu=sum(k*pk for k,pk in enumerate(p))
mu2=sum(k*k*pk for k,pk in enumerate(p))
print('orig mean',mu,'orig second moment',mu2)
# remove k=10
p_new=[pk for k,pk in enumerate(p) if k!=10]
total_new=sum(p_new)
mean_new=sum(k*pk for k,pk in enumerate(p) if k!=10)/total_new
second_new=sum(k*k*pk for k,pk in enumerate(p) if k!=10)/total_new
print('new mean',mean_new,'new second',second_new)
q_new=(second_new-mean_new)/mean_new
print('q_new',q_new)
print('R0_new',q_new)