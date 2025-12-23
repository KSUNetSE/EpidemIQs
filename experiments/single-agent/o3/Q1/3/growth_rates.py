
import math, json
sigma=1/3
gamma=1/5
# moments from previous small networks
beta_er=0.0625 # let's compute again formula
# Actually for small networks: q_er=9.9928, R0=2.5 => beta_er=R0*gamma/q_er
q_er=9.992804317409554
beta_er=2.5*gamma/q_er
q_ba=20.24421052631579
beta_ba=2.5*gamma/q_ba
beta_h=0.5  # homogeneous mixing baseline chosen earlier with R0 2.5

def growth_rate(beta):
    # solve r^2 + (sigma+gamma) r + sigma gamma - beta*sigma = 0
    a=1
    b=sigma+gamma
    c=sigma*gamma - beta*sigma
    disc=b*b -4*a*c
    r=(-b + math.sqrt(disc))/2
    return r
r_h=growth_rate(beta_h)
r_er=growth_rate(beta_er)
r_ba=growth_rate(beta_ba)
print({'r_h':r_h,'r_er':r_er,'r_ba':r_ba})