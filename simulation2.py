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
    def max_positive(self, f):
       # print(f)
       # print([self.U(f[n])-self.C(n) for n in range(self.max_ratings+1)])
        for n in range(self.max_ratings,0,-1):
            if self.U(f[n]) > self.C(n):
                return n
        return 0

def get_user(alpha, c):
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

def sim_metric2(user, F, m, f_gen, N_0, S_0=None, alpha=1, make_plots=False):
    ''' returns the total utility of m users, all with utility and cost functions of user
    uses the constraint function F and decides on f using f_gen. '''
    net_util = 0
    max_n = user.max_ratings
    N, S = N_0, S_0
    contribs = []
    utils = []
    for i in range(m):
        f = f_gen(user, F, N, S, i/m, alpha)
        n_i, util = user.decide(f)
        contribs.append(n_i)
        utils.append(util)
        N += n_i
        S += n_i*n_i
        net_util += util
    if make_plots:
        plt.figure(1)
        plt.plot(range(m), contribs)
        plt.figure(2)
        plt.plot(range(m), utils)
    return net_util, N, S

def lower_n(user, F, N, S):
    return user.get_n([F(n,N,S) for n in range(user.max_ratings+1)])

def upper_n(user, F, N, S):
    return user.max_positive([F(n,N,S) for n in range(user.max_ratings+1)])

def f1(user, F, N, S, r, alpha):
    threshold = int(lower_n(user, F, N, S)*r**alpha+upper_n(user,F,N,S)*(1-r**alpha))
    L = [0]*threshold
    for i in range(user.max_ratings+1-threshold):
        L.append(F(i+threshold, N, S))
    return L

def f2(user, F, N, S, r, alpha):
    if r < alpha:
        threshold = upper_n(user,F,N,S)
    else:
        threshold = lower_n(user, F, N, S)
    L = [0]*threshold
    for i in range(user.max_ratings+1-threshold):
        L.append(F(i+threshold, N, S))
    return L

def read_outfile(fname):
    file = open(fname)
    data = [eval(line) for line in file]
    file.close()
    return data

def make_plots(data, indep_var, line_var, dep_var='utility', restr_map={}):
    if type(data) == str:
        # it's actually a filename, not a data list
        data = read_outfile(data)
    all_vars = list(filter(lambda x: x not in {'utility', 'N', 'S'}, data[0].keys()))
    aux_vars = list(filter(lambda x: x not in {indep_var, line_var}, all_vars))
    D = {x: set() for x in all_vars}
    for line in data:
        for key in line:
            if key in D:
                D[key].add(line[key])
    target = {}
    for var in aux_vars:
        if var in restr_map:
            target[var] = restr_map[var]
        else:
            # restrict it to the median value by default if no restriction is specified
            target[var] = sorted(list(D[var]))[len(D[var])//2]
    filtered = []
    for line in data:
        bad = False
        for var in aux_vars:
            if abs(line[var]-target[var]) > 1e-10:
                bad = True
                break
        if not bad:
            filtered.append(line)
    print('parameter settings: %s' % (target))
    lv_vals = sorted(list(D[line_var]))
    for lv in lv_vals:
        myseries = list(filter(lambda x: abs(x[line_var]-lv) < 1e-10, filtered))
        data = sorted([(x[indep_var], x[dep_var]) for x in myseries], key=lambda x: x[0])
        plt.plot([x[0] for x in data], [x[1] for x in data])
    plt.xlabel(indep_var)
    plt.ylabel(dep_var)
    plt.legend(['%s=%s' % (line_var, lv) for lv in lv_vals])
    plt.show()
        
t = 0.2
# alpha = 1.0
beta = 1.0
gamma = 1.0
# a = 0.0001
b = 0.2
c = 0.0001

u = get_user(2, 0.01)
m = 1000
N_0 = 1000
S_0 = 20000

utils = [0]*6
Nvals = [0]*6
Svals = [0]*6
k = 0

util_main = []

alpha_list = [.2, .6, 1.0, 1.4, 1.8, 2.2]
a_list = [.00001, .00005, .0001, .0005, .001, .01]
'''
# SAM: just realized the problem of two alphas -- I was varying the alpha that goes into the constraint function
# for the alphas in the inner-most loop, what do they represent?
# I'm trying to vary alpha and a and then plot the result on a 3d plot, but I need some help piecing apart what does what?
for alpha in alpha_list:
    for a in a_list:
        print(alpha,a)
        F = get_constraint_func(t,a,b,c,alpha,beta,gamma)
        k = 0
        for f_gen, alpha in [(f2, 0), (f2, .1), (f2, .2), (f1, .1), (f1, .3), (f1, 1)]:
            utils[k], Nvals[k], Svals[k] = sim_metric2(u, F, m, f_gen, N_0, S_0, alpha, make_plots=False)
            print(utils[k], Nvals[k], Svals[k])
            k += 1
        
        util_main.append(utils[:])

print(len(util_main))
'''
# for f_gen, alpha in [(f2, 0), (f2, .1), (f2, .2), (f1, .1), (f1, .3), (f1, 1)]:
#     utils[k], Nvals[k], Svals[k] = sim_metric2(u, F, m, f_gen, N_0, S_0, alpha, make_plots=True)
#     k += 1

# # common labels to all plots
# for j in range(1,3):
#     plt.figure(j)
#     plt.legend(['always max utility', 'max utility after 10%', 'max utility after 20%', 'weighted, exponent .1', 'weighted, exponent .3', 'weighted, exponent 1'])
#     plt.xlabel('user number')
# plt.figure(1)
# plt.ylabel('number of items rated')
# plt.figure(2)
# plt.ylabel('utility')

# plt.figure(3)
# plt.bar(range(6), utils)
# plt.xticks(range(6), ['max after 0%', 'max after 10%', 'max after 20%', 'weighted .1', 'weighted .3', 'weighted 1.0'])
# plt.ylabel('total utility')
# nmin, nmax = 0, 40
# Nmin, Nmax = 0, 2000
# #make_plot(t,alpha,beta,gamma,a,b,c,nmin,nmax,Nmin,Nmax)
# plt.show()
