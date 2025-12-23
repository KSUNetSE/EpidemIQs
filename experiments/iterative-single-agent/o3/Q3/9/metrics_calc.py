
peak_I_deg10=max(res_deg10['I'])
peak_I_random=max(res_random['I'])
final_size_deg10=res_deg10['R'][-1]-len(vacc_nodes_deg10)  # infections + vaccinated or recovered? we set vaccinated nodes to R from start. final R includes vaccinated + recovered from infection
final_size_deg10_filtered=final_size_deg10
final_size_random=res_random['R'][-1]-len(vacc_nodes_random)
print(peak_I_deg10,final_size_deg10_filtered,peak_I_random,final_size_random)