
def q_after(x):
    P1=1/3
    P3=4/7
    P10=2/21
    denom=1 - x*P10
    km=(P1*1 + P3*3 + P10*(1-x)*10)/denom
    k2m=(P1*1 + P3*9 + P10*(1-x)*100)/denom
    return (k2m - km)/km
# find minimal x
xs=[i/1000 for i in range(0,1001)]
thr=None
for x in xs:
    if q_after(x)<=1:
        thr=x
        break
thr