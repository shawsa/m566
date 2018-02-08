import numpy as np
import matplotlib.pyplot as plt
from pytex import *

def gen_A(N):
    n = N**2
    A = np.zeros((n,n))
    A += np.diag(4*np.ones(n), k=0)
    A += np.diag(-1*np.ones(n-1), k=1)
    A += np.diag(-1*np.ones(n-1), k=-1)
    A += np.diag(-1*np.ones(n-N), k=N)
    A += np.diag(-1*np.ones(n-N), k=-N)
    return A

def residual(x,b):
    n = len(x)
    N = int(np.sqrt(n))
    assert N**2 == n
    assert N>= 3
    r = np.zeros(n)
    r[0] = b[0] -4*x[0] + x[1] + x[N]
    for i in range(1, N):
        r[i] = b[i] + x[i-1] - 4*x[i] + x[i+1] + x[i+N]
    for i in range(N, n-N):
        r[i] = b[i] + x[i-N] + x[i-1] - 4*x[i] + x[i+1] + x[i+N]
    for i in range(n-N, n-1):
        r[i] = b[i] + x[i-N] + x[i-1] - 4*x[i] + x[i+1]
    r[-1] = b[-1] + x[-1-N] + x[-2] - 4*x[-1]
    return r

def jacobi_solve(x, b, tol, max_iter=10**9):
    n = len(x)
    N = int(np.sqrt(n))
    assert N**2 == n
    assert N>= 3
    x_old = x
    r = residual(x_old, b)
    res_norms = [np.linalg.norm(r)]
    x_new = x_old + r/4
    iterations = 1
    while np.linalg.norm(r) > np.linalg.norm(b)*tol and iterations < max_iter:
        r = residual(x_new, b)
        res_norms.append(np.linalg.norm(r))
        x_new, x_old = x_new + r/4, x_new
        iterations += 1
    cond = (1+np.cos(np.pi/(N+1)))/(1-np.cos(np.pi/(N+1)))
    return x_new, iterations, cond, res_norms

def p5b(N, tol):
    n = N**2
    x_old = np.ones(n)
    b = np.ones(n)/(N+1)
    r = residual(x_old, b)
    x_new = x_old + r/4
    while np.linalg.norm(r) > np.linalg.norm(b)*tol:
        r = residual(x_new, b)
        x_new, x_old = x_new + r/4, x_new
    print(x_new)
    #test the result by checking the magnitude of the residual
    print(np.linalg.norm(b - np.dot(gen_A(N), x_new)))
    cond = (1+np.cos(np.pi/(N+1)))/(1-np.cos(np.pi/(N+1)))
    print('The condition number is %g.' % cond)

def p5c():
    data = []
    ls = range(2,6)
    Ns = [2**l-1 for l in ls]
    ns = [N**2 for N in Ns]
    iterations = []
    conds = []
    for n, N in zip(ns, Ns):
        x = np.ones(n)
        b = x/(N+1)**2
        solution, it, cond, res_norm = jacobi_solve(x, b, 10**-5)
        iterations.append(it)
        conds.append(cond)
    latex_table([ls, Ns, ns, iterations, conds], ['$l$', '$N$', '$n$', 'iterations required', '$\kappa(A)$'])
    plt.loglog(ns, conds, 'b.')
    plt.xlabel('n')
    plt.ylabel('Condition Number')
    plt.show()

    plt.semilogy(range(1,1+len(res_norm)), res_norm, 'r.')
    plt.xlabel('iteration')
    plt.ylabel('residual norm')
    plt.show()

    


