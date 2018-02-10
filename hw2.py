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
    for i in range(N,n,N):
        A[i,i-1] = 0
        A[i-1,i] = 0
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
    for i in range(N,n,N):
        r[i] -= x[i-1]
    for i in range(N-1,n-1,N):
        r[i] -= x[i+1]
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
    N=3
    tol = 10**-5
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
    errors_at_center = []
    for n, N in zip(ns, Ns):
        x = np.ones(n)
        b = x/(N+1)**2
        solution, it, cond, res_norm = jacobi_solve(x, b, 10**-5)
        #compute exact solution to plot error at center
        exact = np.linalg.solve(gen_A(N), b.reshape((n,1)) )
        errors_at_center.append( '{:0.3e}'.format(float(solution[n//2+1] - exact[n//2+1])) )
        iterations.append(it)
        conds.append(cond)
    latex_table([ls, Ns, ns, iterations, errors_at_center, ['{:0.3e}'.format(cond) for cond in conds]], ['$l$', '$N$', '$n$', 'iterations required', 'error at $(0.5,0.5)$', '$\kappa(A)$'])
    plt.loglog(ns, conds, 'b.')
    plt.xlabel('n')
    plt.ylabel('Condition Number')
    plt.show()

    plt.semilogy(range(1,1+len(res_norm)), res_norm, 'r.')
    plt.xlabel('iteration')
    plt.ylabel('residual norm')
    plt.show()

#************************************************************
def A_mult(x):
    n = len(x)
    N = int(np.sqrt(n))
    assert N**2 == n
    assert N>= 3
    r = np.zeros(n)
    r[0] = 4*x[0] - x[1] - x[N]
    for i in range(1, N):
        r[i] = -x[i-1] + 4*x[i] - x[i+1] - x[i+N]
    for i in range(N, n-N):
        r[i] = -x[i-N] - x[i-1] + 4*x[i] - x[i+1] - x[i+N]
    for i in range(n-N, n-1):
        r[i] = -x[i-N] - x[i-1] + 4*x[i] - x[i+1]
    r[-1] = -x[-1-N] - x[-2] + 4*x[-1]
    for i in range(N,n,N):
        r[i] += x[i-1]
    for i in range(N-1,n-1,N):
        r[i] += x[i+1]
    return r

def cg_solve(x, b, tol, max_iter=10**9):
    n = len(x)
    N = int(np.sqrt(n))
    assert N**2 == n
    assert N>= 3
    x_old = x
    r = residual(x_old, b)
    res_norms = [np.linalg.norm(r)]
    
    iterations = 1
    delta = np.dot(r, r)
    b_delta = np.dot(b,b)
    p = r
    while delta > b_delta * tol**2 and iterations < max_iter:
        s = A_mult(p)
        alpha = delta/np.dot(p,s)
        x_new = x_old + alpha*p
        r -= alpha * s
        res_norms.append(np.linalg.norm(r))
        delta_new = np.dot(r,r)
        p = r + delta_new/delta * p
        x_old, delta = x_new, delta_new
        iterations += 1
    cond = (1+np.cos(np.pi/(N+1)))/(1-np.cos(np.pi/(N+1)))
    return x_new, iterations, cond, res_norms

def p13b():
    N = 3
    n = N**2
    x0 = np.ones(n)
    b = np.ones(n)/(N+1)
    results = cg_solve(x0,b,10**-8)
    print( np.linalg.norm( A_mult(results[0]) - b ) )
    cond = (1+np.cos(np.pi/(N+1)))/(1-np.cos(np.pi/(N+1)))
    print('The condition number is %g.' % cond)

def p13c():
    data = []
    ls = range(2,7)
    Ns = [2**l-1 for l in ls]
    ns = [N**2 for N in Ns]
    iterations = []
    conds = []
    errors_at_center = []
    for n, N in zip(ns, Ns):
        x = np.ones(n)
        b = x/(N+1)**2
        solution, it, cond, res_norm_n = cg_solve(x, b, 10**-5)
        if N == 31:
            res_norm = res_norm_n
        exact = np.linalg.solve(gen_A(N), b.reshape((n,1)) )
        errors_at_center.append( '{:0.3e}'.format(float(solution[n//2+1] - exact[n//2+1])) )
        iterations.append(it)
        conds.append(cond)
    latex_table([ls, Ns, ns, iterations, errors_at_center, ['{:0.3e}'.format(cond) for cond in conds]], ['$l$', '$N$', '$n$', 'iterations required', 'error at $(0.5,0.5)$', '$\kappa(A)$'])
    plt.loglog(ns, conds, 'b.')
    plt.xlabel('n')
    plt.ylabel('Condition Number')
    plt.show()

    plt.semilogy(range(1,1+len(res_norm)), res_norm, 'r.')
    plt.xlabel('iteration')
    plt.ylabel('residual norm')
    plt.show()


#********************************************************
# problem 25

def gen_Aw(N, w, h):
    #h = 1/(N+1)
    n = N**2
    A = np.zeros((n,n))
    A += np.diag((4 - (w*h)**2) *np.ones(n), k=0)
    A += np.diag(-1*np.ones(n-1), k=1)
    A += np.diag(-1*np.ones(n-1), k=-1)
    A += np.diag(-1*np.ones(n-N), k=N)
    A += np.diag(-1*np.ones(n-N), k=-N)
    for i in range(N,n,N):
        A[i,i-1] = 0
        A[i-1,i] = 0
    return A

def Aw_mult(x, w, h):
    n = len(x)
    N = int(np.sqrt(n))
    #h = 1/(N+1)
    diag = (4-(w*h)**2)
    assert N**2 == n
    assert N>= 3
    r = np.zeros(n)
    r[0] = diag*x[0] - x[1] - x[N]
    for i in range(1, N):
        r[i] = -x[i-1] + diag*x[i] - x[i+1] - x[i+N]
    for i in range(N, n-N):
        r[i] = -x[i-N] - x[i-1] + diag*x[i] - x[i+1] - x[i+N]
    for i in range(n-N, n-1):
        r[i] = -x[i-N] - x[i-1] + diag*x[i] - x[i+1]
    r[-1] = -x[-1-N] - x[-2] + diag*x[-1]
    for i in range(N,n,N):
        r[i] += x[i-1]
    for i in range(N-1,n-1,N):
        r[i] += x[i+1]
    return r

def jacobi_relax(x, b, w, h, relax=.8, tol=10**-8, max_iter=10**9):
    n = len(x)
    N = int(np.sqrt(n))
    diag = (4-(w*h)**2)
    assert N**2 == n
    assert N>= 3
    x_new = x
    x_old = x_new
    iterations = 1
    r = b - Aw_mult(x_new, w, h)
    res_norms = [np.linalg.norm(r)]
    x_new = x_new + r/diag
    x_new = relax*x_new + (1-relax)*x_old
    x_old = x_new 
    while np.linalg.norm(r) > np.linalg.norm(b)*tol and iterations < max_iter:
        x_old = x_new
        r = b - Aw_mult(x_new, w, h)
        res_norms.append(np.linalg.norm(r))
        x_new = x_old + relax*r/diag
        x_old = x_new
        iterations += 1
    return x_new, iterations, res_norms

def prolongate(x):
    n = len(x)
    N = int(np.sqrt(n))
    grid_c = x.reshape((N,N))
    
    grid = np.zeros((2*N - 1,2*N - 1))
    for i in range(N-1):
        for j in range(N-1):
            grid[2*i, 2*j] = grid_c[i,j]
            grid[2*i+1, 2*j] = (grid_c[i,j] + grid_c[i+1,j])*.5
            grid[2*i, 2*j+1] = (grid_c[i,j] + grid_c[i,j+1])*.5
            grid[2*i+1, 2*j+1] = (grid_c[i,j] + grid_c[i+1,j] + grid_c[i,j+1] + grid_c[i+1,j+1])*.25
        grid[-1, 2*i] = grid_c[-1,i]
        grid[-1, 2*i+1] = (grid_c[-1,i] + grid_c[-1,i+1])*.5
        grid[2*i, -1] = grid_c[i, -1]
        grid[2*i+1, -1] = (grid_c[i, -1] + grid_c[i+1, -1])*.5
    grid[-1,-1] = grid_c[-1,-1]
    return grid.ravel()


def test():
    N = 2**7 - 1
    n = N**2
    h = 1/(N+1)
    w = 1
    assert (w*h)**2 < 4 - 4 * np.cos(h*np.pi)
    x = np.ones(n)
    b = x*h
    sol = jacobi_relax(x, b, w, h)[0]
    print(np.linalg.norm(b - Aw_mult(sol,w,h)) )
    

