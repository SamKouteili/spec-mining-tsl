'''
Compositional Function Example
Spec is (x+2) *2 is a funciton. WTS discovery of compositional functions
'''

VARS = {
    "X": "int",
}

FUNCTIONS = {
    "Plus2": ("int->int", lambda X: X + 2),
    "Mult2":("int->int", lambda X: X * 2)
}

PREDICATES = {

}


