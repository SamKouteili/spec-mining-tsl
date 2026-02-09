VARS = {
    "goal[0]": "const int",
    "goal[1]": "const int",
    "hole0[0]": "const int",
    "hole0[1]": "const int",
    "hole1[0]": "const int",
    "hole1[1]": "const int",
    "hole2[0]": "const int",
    "hole2[1]": "const int",
    "player[0]": "int",
    "player[1]": "const int",
}

FUNCTIONS = {
    "inc1": ("int->int", lambda x: (x + 1), "add {0} i1()"),
}

PREDICATES = {
    "eq": ("tuple2->tuple2->bool", lambda x, y: x == y),
    "eqC": ("tuple2->const tuple2->bool", lambda x, y: x == y),
}

# Tuple variable metadata
TUPLE_VARS = {
    "player": {"arity": 2, "const": False},
    "goal": {"arity": 2, "const": True},
    "hole0": {"arity": 2, "const": True},
    "hole1": {"arity": 2, "const": True},
    "hole2": {"arity": 2, "const": True},
}
