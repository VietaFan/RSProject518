import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class User(object):
    def __init__(self, util_func, cost_func, inv_util_func=None):
        ''' assumptions: util_func : [0,1] -> [0,1] is strictly increasing,
        cost_func : N -> R is increasing, util_func(0) = 0, cost_func(0) = 0'''
        self.U = util_func
        self.C = cost_func
        self.Uinv = inv_util_func
        self.max_ratings = 0 # max number of ratings this user can possibly provide
        while self.C(self.max_ratings) < self.U(1.0):
            self.max_ratings += 1
        self.max_ratings -= 1
    def decide(self, f):
        ''' f : list of prediction accuracies of length self.max_ratings()+1
        returns (n*, u), where n* = argmax_n U(f(n))-C(n) and u = U(f(n*))-c(n*)'''
        max_util = 0.0
        best_n = 0
        for n in range(self.max_ratings+1):
            util = self.U(f[n])-self.C(n)
            if util > max_util:
                best_n, max_util = n, util
        return best_n, max_util
    def get_n(self, f):
        return self.decide(f)[0]
    def get_util(self, f):
        return self.decide(f)[1]

def make_user(alpha, c):
    ''' default user type: utility function U(x) = x**alpha, cost function C(x) = cx '''
    return User(lambda x: x**alpha, lambda x: c*x, lambda x: x**(1.0/alpha))

def get_constraint_func(t, a, b, c, alpha=1.0, beta=1.0, gamma=1.0, use_S=True):
    g = lambda a, alpha, x: ((a*x)**alpha)/(1.0+(a*x)**alpha)
    if use_S:
        return lambda n, N, S: t*g(a,alpha,N)+(1.0-t)*g(b,beta,n)*g(c,gamma,S-N)
    else:
        return lambda n, N: t*g(a,alpha,N)+(1.0-t)*g(b,beta,n)*g(c,gamma,N)
    
def rand_user(userlist, probs):
    r = random.random()
    for i in range(len(probs)):
        r -= probs[i]
        if r < 0:
            return userlist[i]
    raise Exception('sum(probs) < 1')

def make_plot(t,alpha,beta,gamma,a,b,c,nmin,nmax,Nmin,Nmax):
    F = get_constraint_func(t,a,b,c,alpha,beta,gamma,use_S=False)
    A = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            A[99-i,j] = F(i*(nmax-nmin)/100+nmin,j*(Nmax-Nmin)/100+Nmin)
    plt.imshow(A, interpolation='none', extent=[Nmin,Nmax,nmin,nmax], aspect='auto')
    plt.colorbar()
    plt.ylabel('n: number of ratings given by user')
    plt.xlabel('N: number of ratings in system database')
    plt.title('F(n,N): t = %s, a = %s, b = %s, alpha = %s, beta = %s' % (t,a,b,alpha,beta))
    plt.show()

t = 0.2
alpha = 0.5
beta = 2.0
gamma = 2.0
a = 0.001
b = 0.2
c = 0.003

nmin, nmax = 0, 40
Nmin, Nmax = 0, 2000
make_plot(t,alpha,beta,gamma,a,b,c,nmin,nmax,Nmin,Nmax)

