
# For full compatibility, break parameter sweeps as a list of dicts for each scenario, plus a baseline scenario as a list (1 element)
scan_parameters_list = [
    {"beta": 0.15, "gamma": 0.08, "xi": 0.005},
    {"beta": 0.15, "gamma": 0.08, "xi": 0.015},
    {"beta": 0.15, "gamma": 0.15, "xi": 0.005},
    {"beta": 0.15, "gamma": 0.15, "xi": 0.015},
    {"beta": 0.3,  "gamma": 0.08, "xi": 0.005},
    {"beta": 0.3,  "gamma": 0.08, "xi": 0.015},
    {"beta": 0.3,  "gamma": 0.15, "xi": 0.005},
    {"beta": 0.3,  "gamma": 0.15, "xi": 0.015},
    {"beta": 0.5,  "gamma": 0.08, "xi": 0.005},
    {"beta": 0.5,  "gamma": 0.08, "xi": 0.015},
    {"beta": 0.5,  "gamma": 0.15, "xi": 0.005},
    {"beta": 0.5,  "gamma": 0.15, "xi": 0.015}
]
baseline_parameters_list = [{"beta": 0.3, "gamma": 0.1, "xi": 0.01}]

result_list_fmt = {
    "parameters": scan_parameters_list + baseline_parameters_list,
    "initial_condition_desc": [
        "Random selection: 99% Unaware, 1% Posting, 0% Fatigued (per routine viral trend seeding; no initial Fatigued)"
    ],
    "initial_conditions": [{"U": 99, "P": 1, "F": 0}]
}
result_list_fmt