VARS = {
    "count": "int",
    "isWeakDealer": "const bool",
    "standThreshold": "const int",
    "standVsWeakMin": "const int",
    "stood": "bool",
}

FUNCTIONS = {
    # No functions synthesized
}

PREDICATES = {
    "eq": ("int->int->bool", lambda x, y: x == y),
    "eqC": ("int->const int->bool", lambda x, y: x == y),
    "lt": ("int->int->bool", lambda x, y: x < y),
    "ltC": ("int->const int->bool", lambda x, y: x < y),
}

# Boolean stream variables (treated as predicates, not updates)
BOOLEAN_VARS = [
    "isWeakDealer",
    "stood",
]
