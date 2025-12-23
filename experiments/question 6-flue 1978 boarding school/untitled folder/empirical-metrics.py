
# Empirical peak value and timing for B(t)
emp_peak_B_value = empirical_B.max()
emp_peak_B_time = empirical_days[empirical_B.argmax()]

N = 763  # population size assumed same

# Final empirical cumulative Attack Rate = fraction ever recorded as B or C by end
# Assuming cumulative is max value of (B+C) over time divided by N
emp_cumulative_cases = (empirical_B + empirical_C).max()
emp_final_AR = emp_cumulative_cases / N

# Main epidemic duration: days from first to last nonzero B or C
nonzero_BC_indices = np.where((empirical_B > 0) | (empirical_C > 0))[0]
if len(nonzero_BC_indices) > 0:
    emp_first_day = empirical_days[nonzero_BC_indices[0]]
    emp_last_day = empirical_days[nonzero_BC_indices[-1]]
    emp_epidemic_duration = emp_last_day - emp_first_day
else:
    emp_first_day = None
    emp_last_day = None
    emp_epidemic_duration = 0

emp_peak_B_value, emp_peak_B_time, emp_final_AR, emp_epidemic_duration, emp_first_day, emp_last_day