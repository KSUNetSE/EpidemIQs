
# As final summary, provide mapping and diagnostic csv for scenario/parameter indices to guarantee traceability.
import pandas as pd
beta_list = [0.15, 0.15, 0.15, 0.15, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.3]
gamma_list = [0.08, 0.08, 0.15, 0.15, 0.08, 0.08, 0.15, 0.15, 0.08, 0.08, 0.15, 0.15, 0.1]
xi_list = [0.005, 0.015, 0.005, 0.015, 0.005, 0.015, 0.005, 0.015, 0.005, 0.015, 0.005, 0.015, 0.01]
index_list = list(range(13))
df = pd.DataFrame({'index': index_list, 'beta': beta_list, 'gamma': gamma_list, 'xi': xi_list, 'results_csv': [f'results-{str(j).zfill(2)}.csv' for j in index_list], 'results_fig': [f'results-{str(j).zfill(2)}.png' for j in index_list ]})
mapping_path = os.path.join(os.getcwd(), 'output', 'scenario_parameter_map.csv')
df.to_csv(mapping_path, index=False)
mapping_path
