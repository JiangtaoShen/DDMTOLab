"""Fallback problem parameters for the GUI.

These tables are only consulted for a suite the scanner could not find
(``utils.problem_scanner.is_known_problem_suite`` returns False) - for every
suite shipped with the platform the scanner reads the real signatures instead.
Keys are therefore suite *class* names, the same identifiers the scanner
reports.
"""

# Problem-specific parameters (fallback when auto-scan fails)
# Only type and default needed, description = param name
PROBLEM_PARAMS = {
    "ZDT": {"D": {"type": "int", "default": 30}},
    # DTLZ and WFG derive D from M when it is not given, so there is no fixed
    # signature default; the platform default (50) is what the scanner reports.
    "DTLZ": {"M": {"type": "int", "default": 3}, "D": {"type": "int", "default": 50}},
    "WFG": {"M": {"type": "int", "default": 3}, "Kp": {"type": "int", "default": 4},
            "D": {"type": "int", "default": 50}},
    "UF": {"D": {"type": "int", "default": 30}},
    "CF": {"M": {"type": "int", "default": 2}, "D": {"type": "int", "default": 10}},
    "MW": {"M": {"type": "int", "default": 2}, "D": {"type": "int", "default": 15}},
    "MTMO_DTLZ": {"M": {"type": "int", "default": 3}, "D": {"type": "int", "default": 10}},
    "CEC10_CSO": {"D": {"type": "int", "default": 10}},
    "CLASSICALSO": {"D": {"type": "int", "default": 50}},
    "STSOtest": {"D": {"type": "int", "default": 50}},
    "CEC19MaTSO": {"K": {"type": "int", "default": 10}},
    "STOP": {"K": {"type": "int", "default": 10}},
    "CMT": {"D": {"type": "int", "default": 50}},
    "CEC19_MaTMO": {"K": {"type": "int", "default": 10}},
    "MO_SCP": {"K": {"type": "int", "default": 5}},
    "PKACP": {"K": {"type": "int", "default": 20}, "D": {"type": "int", "default": 20}},
}

# Suites with fixed dimensions (no D parameter)
FIXED_DIMENSION_SUITES = [
    "CEC17MTSO", "CEC17MTSO_10D", "CEC19MaTSO", "STOP", "ManyTask_10D",
    "CEC17MTMO", "CEC19MTMO", "CEC19_MaTMO", "CEC21MTMO", "MTMOInstances",
    "MO_SCP", "NN_Training", "PEPVM", "PINN_HPO", "SCP", "SOPM", "TSP",
]

# Suites with fixed objectives (no M parameter)
# CF: CF1-CF7=2obj, CF8-CF10=3obj; ZDT: all 2obj; UF: UF1-7=2obj, UF8-10=3obj;
# MW: fixed per problem. M stays configurable only for DTLZ, WFG and MTMO_DTLZ.
FIXED_OBJECTIVES_SUITES = ["ZDT", "CF", "UF", "MW"]
