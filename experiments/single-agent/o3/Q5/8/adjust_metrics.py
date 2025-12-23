
metrics_adj=[]
for m in metrics:
    inf = m['final_size'] - m['vaccinated_prop']
    m['infection_final_size']=inf
    metrics_adj.append(m)
metrics_adj