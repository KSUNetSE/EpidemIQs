
# Fix for output formatting: flatten parameter sweeps and baseline for compatibility
# Output dict[str,list[float]] for parameter scan (not as list of dicts)
scan_parameters = {
    "beta": [0.15, 0.15, 0.15, 0.15, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5],
    "gamma": [0.08, 0.08, 0.15, 0.15, 0.08, 0.08, 0.15, 0.15, 0.08, 0.08, 0.15, 0.15],
    "xi":   [0.005,0.015,0.005,0.015,0.005,0.015,0.005,0.015,0.005,0.015,0.005,0.015]
}
baseline_parameters = {"beta": 0.3, "gamma": 0.1, "xi": 0.01}

result_fmt = {
    "parameters": {"scan": scan_parameters, "baseline": baseline_parameters},
    "initial_condition_desc": [
        "Random selection: 99% Unaware, 1% Posting, 0% Fatigued (per routine viral trend seeding; no initial Fatigued)"
    ],
    "initial_conditions": [{"U": 99, "P": 1, "F": 0}]
}
result_fmt