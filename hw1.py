import numpy as np
import matplotlib.pyplot as plt

#code for problem 4
def p4():
    A = np.array([[1, 0],[1,1],[1,2]])
    b = np.array([[.1],[.9],[2]])
    x = np.linalg.lstsq(A,b)[0]
    print(A)
    print(b)
    print(x)

    g1 = np.exp(x[0])
    g2 = x[1]

    print('gamma_1 = {}'.format(g1))
    print('gamma_2 = {}'.format(g2))

    ts = np.linspace(-.1,2.1,1000)
    us = g1*np.exp(g2*ts)
    
    plt.plot([0,1,2], np.exp(b), 'go')
    plt.plot(ts, us, 'b-')
    plt.show()
