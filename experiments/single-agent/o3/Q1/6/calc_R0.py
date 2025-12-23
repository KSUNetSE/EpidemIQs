
import numpy as np,json
mk_er=10.0376
k2_er=110.8
mk_ba=9.99
k2_ba=272.5716
beta=0.03
gamma=1/7
R0_hom=beta/gamma
R0_er=beta/gamma*(k2_er-mk_er)/mk_er
R0_ba=beta/gamma*(k2_ba-mk_ba)/mk_ba
json_out={'R0_hom':R0_hom,'R0_er':R0_er,'R0_ba':R0_ba}
