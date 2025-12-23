
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('output/results-12.csv')
# Show a table of S, I, R at key timepoints: start, peak, near end
start = df.iloc[0]
peak_idx = df['I'].idxmax()
peak = df.iloc[peak_idx]
end = df.iloc[-1]
report_table = pd.DataFrame([start, peak, end], index=["Start", "Peak", "Final"])[["time", "S", "I", "R"]]
report_table
