
# The required format is: parameters: Dict[str, List[float]] | Dict[str, float]
# So for grid scan: each key (rate name) maps to a List[float].
scan_parameters_flat = {
    "beta": [0.15, 0.15, 0.15, 0.15, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.3],
    "gamma": [0.08, 0.08, 0.15, 0.15, 0.08, 0.08, 0.15, 0.15, 0.08, 0.08, 0.15, 0.15, 0.1],
    "xi":   [0.005,0.015,0.005,0.015,0.005,0.015,0.005,0.015,0.005,0.015,0.005,0.015,0.01]
}

result_scan_flat = {
    "parameters": scan_parameters_flat,
    "initial_condition_desc": [
        "Random selection: 99% Unaware, 1% Posting, 0% Fatigued (per routine viral trend seeding; no initial Fatigued)"
    ],
    "initial_conditions": [{"U": 99, "P": 1, "F": 0}]
}
result_scan_flat