
# Check whether each set of parameters yields the intended R0 (network-effective) for each scenario
# Theoretical: beta_network * q / gamma == R0
R0_er_hi_check = params_all['ER_R0_gt1']['beta'] * q_values['q_er'] / params_all['ER_R0_gt1']['gamma']
R0_er_lo_check = params_all['ER_R0_lt1']['beta'] * q_values['q_er'] / params_all['ER_R0_lt1']['gamma']
R0_ba_hi_check = params_all['BA_R0_gt1']['beta'] * q_values['q_ba'] / params_all['BA_R0_gt1']['gamma']
R0_ba_lo_check = params_all['BA_R0_lt1']['beta'] * q_values['q_ba'] / params_all['BA_R0_lt1']['gamma']

checks = {
    'ER_R0_gt1': R0_er_hi_check,
    'ER_R0_lt1': R0_er_lo_check,
    'BA_R0_gt1': R0_ba_hi_check,
    'BA_R0_lt1': R0_ba_lo_check
}
checks