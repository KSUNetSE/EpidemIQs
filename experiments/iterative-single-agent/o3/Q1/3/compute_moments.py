
import networkx as nx, numpy as np, os, scipy.sparse as sparse, json, math, random, matplotlib.pyplot as plt
N=5000
k_avg=10
# ER network
p = k_avg/(N-1)
G_er = nx.fast_gnp_random_graph(N, p, seed=42)
# BA network
m = k_avg//2
G_ba = nx.barabasi_albert_graph(N, m, seed=42)
# compute moments
import numpy as np
k_er = np.array([d for n,d in G_er.degree()])
mean_er = k_er.mean()
second_er = (k_er**2).mean()
q_er = (second_er - mean_er)/mean_er
k_ba = np.array([d for n,d in G_ba.degree()])
mean_ba = k_ba.mean()
second_ba = (k_ba**2).mean()
q_ba = (second_ba - mean_ba)/mean_ba
q_er, q_ba, mean_er, mean_ba, second_er, second_ba