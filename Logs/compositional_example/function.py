'''
Compositional Function Example
Spec is (x+2) *2 is a funciton. WTS discovery of compositional functions
'''

VARS = {
    "X": int,
}

FUNCTIONS = {
    "Plus": ("int->int", lambda X: X + 2),
    "Multiplied":("int->int", lambda X: X * 2)
}


