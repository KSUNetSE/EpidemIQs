
# Correcting the calculation of first and last days when B or C > 1

# Boolean mask for when B or C > 1
threshold = 1
condition = (B > threshold) | (C > threshold)

if condition.any():
    first_index = condition.idxmax()  # index of first True
    last_index = condition[::-1].idxmax()  # index of last True by reversed series
    first_day_corrected = time.iloc[first_index]
    last_day_corrected = time.iloc[last_index]
    epidemic_duration_corrected = last_day_corrected - first_day_corrected
else:
    first_day_corrected = None
    last_day_corrected = None
    epidemic_duration_corrected = 0

first_day_corrected, last_day_corrected, epidemic_duration_corrected