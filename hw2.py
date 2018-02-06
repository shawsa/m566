import numpy as np

def gen_A(n):
    N = n**2
    A = np.zeros((N,N))
    A += np.diag(4*np.ones(N), k=0)
    A += np.diag(-1*np.ones(N-1), k=1)
    A += np.diag(-1*np.ones(N-1), k=-1)
    A += np.diag(-1*np.ones(N-n), k=n)
    A += np.diag(-1*np.ones(N-n), k=-n)
    return A
