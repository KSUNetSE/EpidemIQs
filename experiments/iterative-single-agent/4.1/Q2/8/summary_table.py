
# Summarize and prepare table for reporting
with open("output/sir_parameters.json") as f:
    params = json.load(f)
with open("output/metrics-11.csv") as f:
    import csv
    reader = csv.DictReader(f)
    for row in reader:
        metrics = row
# Construct table from scenario and metrics
summary_table = {
    "R0": params['R0'],
    "beta": round(params['beta'], 5),
    "gamma": round(params['recovery_rate'], 5),
    "Network mean degree": round(params['network']['mean_deg'], 2),
    "Network mean degree sq": round(params['network']['mean_deg2'], 2),
    "# nodes": params['network']['nodes'],
    "Initial Infected": params['init_cond']['I'],
    "Epidemic Duration": round(float(metrics['Epidemic_Duration']),2),
    "Peak Infected": int(metrics['Peak_Infected']),
    "Peak Time": round(float(metrics['Peak_Time']),2),
    "Final Recovered": int(metrics['Final_Recovered']),
    "Doubling Time": round(float(metrics['Doubling_Time']),2)
}
import pandas as pd
pd.DataFrame([summary_table]).to_csv("output/summary_table.csv", index=False)
summary_table