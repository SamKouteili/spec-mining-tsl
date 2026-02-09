VARS = {
    "count": "int",
    "dealer": "const int",
    "standThreshold": "const int",
    "stood": "bool",
}

FUNCTIONS = {
    "add": ("int->int", lambda x: (x + (1 + (1 + 1))), "add {0} (add i1() (add i1() i1()))"),
    "add_2": ("int->int", lambda x: (x + (1 + 1)), "add {0} (add i1() i1())"),
    "add_3": ("int->int", lambda x: (x + (1 + (1 + (1 + (1 + (1 + (1 + (1 + (1 + (1 + 1)))))))))), "add {0} (add i1() (add i1() (add i1() (add i1() (add i1() (add i1() (add i1() (add i1() (add i1() i1())))))))))"),
    "add_4": ("int->int", lambda x: (x + (1 + (1 + (1 + (1 + (1 + 1)))))), "add {0} (add i1() (add i1() (add i1() (add i1() (add i1() i1())))))"),
    "add_5": ("int->int", lambda x: (x + (1 + (1 + (1 + 1)))), "add {0} (add i1() (add i1() (add i1() i1())))"),
    "add_6": ("int->int", lambda x: ((x + x) + 1), "add (add {0} {0}) i1()"),
    "add_7": ("int->int", lambda x: (x + (1 + (1 + (1 + (1 + (1 + (1 + 1))))))), "add {0} (add i1() (add i1() (add i1() (add i1() (add i1() (add i1() i1()))))))"),
    "dec1": ("int->int", lambda x: (x - 1), "sub {0} i1()"),
    "inc1": ("int->int", lambda x: (x + 1), "add {0} i1()"),
    "sub": ("int->int", lambda x: ((x + x) - (1 + (1 + (1 + (1 + (1 + (1 + 1))))))), "sub (add {0} {0}) (add i1() (add i1() (add i1() (add i1() (add i1() (add i1() i1()))))))"),
    "sub_2": ("int->int", lambda x: (x - (1 + (1 + 1))), "sub {0} (add i1() (add i1() i1()))"),
    "sub_3": ("int->int", lambda x: ((x + x) - (1 + (1 + (1 + 1)))), "sub (add {0} {0}) (add i1() (add i1() (add i1() i1())))"),
    "sub_4": ("int->int", lambda x: ((x + x) - (1 + (1 + (1 + (1 + (1 + 1)))))), "sub (add {0} {0}) (add i1() (add i1() (add i1() (add i1() (add i1() i1())))))"),
    "sub_5": ("int->int", lambda x: ((x + x) - 1), "sub (add {0} {0}) i1()"),
    "sub_6": ("int->int", lambda x: ((x + (x + x)) - 1), "sub (add {0} (add {0} {0})) i1()"),
    "sub_7": ("int->int", lambda x: ((x + x) - (1 + (1 + 1))), "sub (add {0} {0}) (add i1() (add i1() i1()))"),
}

PREDICATES = {
    "eq": ("int->int->bool", lambda x, y: x == y),
    "eqC": ("int->const int->bool", lambda x, y: x == y),
    "lt": ("int->int->bool", lambda x, y: x < y),
    "ltC": ("int->const int->bool", lambda x, y: x < y),
}

# Boolean stream variables (treated as predicates, not updates)
BOOLEAN_VARS = [
    "stood",
]
