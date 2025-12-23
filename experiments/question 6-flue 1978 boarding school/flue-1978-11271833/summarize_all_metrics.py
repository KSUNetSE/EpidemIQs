
# Repeat key extractions for all other metric files like 12 to 16
metrics_summary = {}
for file in files:
    df = data_dict[file].set_index('Unnamed: 0')
    final_ar = df.loc['final_attack_rate','0']
    R0 = df.loc['R0','0']
    mse_B = df.loc['mse_B','0']
    S_final = df.loc['S_final','0']
    N = df.loc['N','0']
    computed_ar = 1 - S_final/N
    metrics_summary[file] = {
        'final_attack_rate': final_ar,
        'R0': R0,
        'mse_B': mse_B,
        'S_final': S_final,
        'N': N,
        'computed_ar': computed_ar
    }
metrics_summary