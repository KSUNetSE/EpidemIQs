
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error

def eval_metrics(rescsv, tag):
    df = pd.read_csv(rescsv)
    case_xlsx = os.path.join(os.getcwd(), "output", "cases_data.xlsx")
    try:
        obs_data = pd.read_excel(case_xlsx)
        obs_B = obs_data['B']
        obs_t = obs_data['time'] if 'time' in obs_data else np.arange(len(obs_B))
    except Exception as e:
        obs_B = np.zeros(len(df['time']))
        obs_t = df['time']
    pred_t = df['time']
    pred_B = df['B']
    if len(obs_B) != len(pred_B):
        from scipy.interpolate import interp1d
        fB = interp1d(pred_t, pred_B, kind='linear', fill_value="extrapolate")
        pred_B_obsgrid = fB(obs_t)
    else:
        pred_B_obsgrid = pred_B
    mse = mean_squared_error(obs_B, pred_B_obsgrid)
    S_final = df['S'].iloc[-1]
    AR = 1 - S_final / 763
    R0 = 0.1365451943519671 / 1.1111111111111112
    metrics = {
        'final_attack_rate': AR,
        'R0': R0,
        'mse_B': mse, 
        'S_final': S_final,
        'N': 763,
        'beta': 0.1365451943519671,
        'gamma': 1.1111111111111112
    }
    cap = f"{tag}: AR={AR:.3f}, R0={R0:.2f}, MSE_B={mse:.3f}"
    pd.Series(metrics).to_csv(rescsv.replace('.csv', '-metrics.csv'))
    with open(rescsv.replace('.csv', '-metrics.txt'), 'w') as f:
        f.write(cap)
# beta-10%
eval_metrics(os.path.join(os.getcwd(), 'output', 'results-12.csv'), tag='SBM, beta-10%')
# beta+10%
eval_metrics(os.path.join(os.getcwd(), 'output', 'results-13.csv'), tag='SBM, beta+10%')
# long latent (sigma down)
eval_metrics(os.path.join(os.getcwd(), 'output', 'results-14.csv'), tag='SBM, sigma=0.7 (+0.5d Tgen)')
# 3 I in 3 houses
eval_metrics(os.path.join(os.getcwd(), 'output', 'results-15.csv'), tag='SBM seed 3 in 3 houses')
# ER network
eval_metrics(os.path.join(os.getcwd(), 'output', 'results-16.csv'), tag='ER baseline')
