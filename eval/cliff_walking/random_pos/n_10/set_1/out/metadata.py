VARS = {
    "cliffHeight": "const int",
    "cliffXMax": "const int",
    "cliffXMin": "const int",
    "goalX": "const int",
    "goalY": "const int",
    "x": "int",
    "y": "int",
}

FUNCTIONS = {
    "dec1": ("int->int", lambda x: (x - 1), "sub {0} i1()"),
    "inc1": ("int->int", lambda x: (x + 1), "add {0} i1()"),
}

PREDICATES = {
    "eq": ("int->int->bool", lambda x, y: x == y),
    "eqC": ("int->const int->bool", lambda x, y: x == y),
    "lt": ("int->int->bool", lambda x, y: x < y),
    "ltC": ("int->const int->bool", lambda x, y: x < y),
}
