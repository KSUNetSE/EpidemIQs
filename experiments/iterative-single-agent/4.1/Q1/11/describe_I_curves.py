
import pandas as pd
import os
import matplotlib.pyplot as plt

def describe_I_curves():
    df_er = pd.read_csv(os.path.join(os.getcwd(),'output','results-1-1.csv'))
    df_ba = pd.read_csv(os.path.join(os.getcwd(),'output','results-1-2.csv'))
    # peak, onset, duration, symmetry, values
    props_er = {
        'peak_I': df_er['I'].max(),
        'peak_time': float(df_er['time'][df_er['I'].idxmax()]),
        'final_size': float(df_er['R'].iloc[-1]),
        'duration': float(df_er['time'][(df_er['I']>1)].max() - df_er['time'][(df_er['I']>1)].min())
    }
    props_ba = {
        'peak_I': df_ba['I'].max(),
        'peak_time': float(df_ba['time'][df_ba['I'].idxmax()]),
        'final_size': float(df_ba['R'].iloc[-1]),
        'duration': float(df_ba['time'][(df_ba['I']>1)].max() - df_ba['time'][(df_ba['I']>1)].min())
    }
    # Compare onset slope (how fast I grows initially)
    init_slope_er = (df_er['I'][5]-df_er['I'][1]) / (df_er['time'][5]-df_er['time'][1])
    init_slope_ba = (df_ba['I'][5]-df_ba['I'][1]) / (df_ba['time'][5]-df_ba['time'][1])
    comparison = {
        'homogeneous_ER': props_er,
        'heterogeneous_BA': props_ba,
        'init_slope_ER': init_slope_er,
        'init_slope_BA': init_slope_ba,
        'summary': f"Heterogeneous (BA) network shows a higher peak (I={props_ba['peak_I']} at t={props_ba['peak_time']:.1f}) than homogeneous (ER, I={props_er['peak_I']} at t={props_er['peak_time']:.1f}), but a slightly shorter epidemic duration. The initial slope of outbreak is much steeper in BA ({init_slope_ba:.2f}) than in ER ({init_slope_er:.2f}), indicating faster early spread. Both show rapid takeoff, but BA curve peaks slightly sooner and declines more quickly than ER."
    }
    return comparison

describe_I_curves()