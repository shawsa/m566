import numpy as np

def poly_least_squares(t, y, deg):
    m = len(t)
    A = np.ones( (m,1) )
    
    t_row = t.reshape( (m,1) )
    row = np.ones( (m, 1) )
    for i in range(deg):
        row *= t_row
        A = np.concatenate( (A, row), axis=0)

    print(A)
