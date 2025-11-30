VARS = {
    "ball": "int"
}

FUNCTIONS = {
    "moveRight": ("int->int", lambda ball: ball + 1),
    "moveLeft": ("int->int", lambda ball: ball - 1),
}

PREDICATES = {
    "rightMost": ("int->bool", lambda ball: ball == -2),
    "leftMost": ("int->bool", lambda ball: ball == 2),
}

