
# Select last 20 time points for final R to check CIs for quantile approximation
last_20_R = data[['time', 'SIR_R', 'SIR_R_90ci_lower', 'SIR_R_90ci_upper']].tail(20)

# Simple quantile approximation: using linear interpolation between bounds
# Use lower CI as approx 5% quantile, upper CI as 95% quantile
quantiles_final_R = {
    '5%': last_20_R['SIR_R_90ci_lower'].min(),
    '50%_approx': last_20_R['SIR_R'].median(),
    '95%': last_20_R['SIR_R_90ci_upper'].max()
}

quantiles_final_R