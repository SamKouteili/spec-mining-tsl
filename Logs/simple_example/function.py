'''
Ball moving in a 1D plane. must eventually move to the min or max if at one must eventually move away from it.
Must always make monotonic progress to the other bound if it has reached the other bound 
'''
VARS = {
    "ball": int
}

FUNCTIONS = {
    "moveRight": ("int->int", lambda ball: ball + 1),
    "moveLeft": ("int->int", lambda ball: ball - 1),
}

PREDICATES = {
    "rightMost": ("int->bool", lambda ball: ball == -2),
    "leftMost": ("int->bool", lambda ball: ball == 2),
}