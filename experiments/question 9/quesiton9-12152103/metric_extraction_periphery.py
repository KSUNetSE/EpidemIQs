
peak_D_idx_periphery = df_periphery['D'].idxmax()
peak_D_periphery = df_periphery.loc[peak_D_idx_periphery, 'D']
peak_time_periphery = df_periphery.loc[peak_D_idx_periphery, 'time']
peak_D_lower_periphery = df_periphery.loc[peak_D_idx_periphery, 'D_90ci_lower']
peak_D_upper_periphery = df_periphery.loc[peak_D_idx_periphery, 'D_90ci_upper']

final_L_periphery = df_periphery.loc[df_periphery.index[-1], 'L']
final_L_lower_periphery = df_periphery.loc[df_periphery.index[-1], 'L_90ci_lower']
final_L_upper_periphery = df_periphery.loc[df_periphery.index[-1], 'L_90ci_upper']

post_peak_D_periphery = df_periphery.loc[peak_D_idx_periphery:, 'D']
min_D_idx_periphery = post_peak_D_periphery.idxmin()
crisis_duration_periphery = df_periphery.loc[min_D_idx_periphery, 'time'] - df_periphery.loc[0, 'time']

area_D_periphery = np.trapz(df_periphery['D'], df_periphery['time'])

(peak_D_periphery, peak_time_periphery, final_L_periphery, crisis_duration_periphery, area_D_periphery, peak_D_lower_periphery, peak_D_upper_periphery, final_L_lower_periphery, final_L_upper_periphery)