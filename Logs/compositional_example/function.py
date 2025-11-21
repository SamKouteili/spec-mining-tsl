'''
Compositional Function Example
Spec: [x <- *2 *2 x] & X (G [x <- +2 x]) 
    WTS discovery of compositional functions
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


