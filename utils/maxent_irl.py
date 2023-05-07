import numpy as np
from copy import deepcopy
from scipy.optimize import minimize

def softmax(X):
    dim = len(X.shape)
    if dim == 1:
        X = np.sort(X)
        return X[-1] + np.log(1 + np.sum(np.exp(X[:-1] - X[-1])))
    elif dim > 1:
        X = np.sort(X,axis=0)
        return X[-1,:] + np.log(1 + np.sum(np.exp(X[:-1,:] - X[-1,:]),axis=0))
        
def compute_pt_lamb(lamb,Tmat,p0,gam,T):
    S = Tmat.shape[0]
    A = Tmat.shape[1]
    
    log_alph = np.ones((T+1,S,A)) 
    log_beta = np.ones((T+1,S,A))
    
    #print(p0.shape,lamb.shape,log_alph[0].shape)

    log_alph[0] = np.log(p0)[:,None] + lamb
    log_beta[0] = (gam**T)*lamb

    Tmat_P = Tmat.reshape((-1,S))
    Tmat_C = Tmat.transpose(2,0,1)
    Tmat_C = Tmat_C[:,None,:,:]

    for t in range(1,T+1):
        log_alph[t] = (gam**t)*lamb + (softmax(np.log(Tmat_P) + log_alph[t-1].flatten()[:,None]))[:,None]
        temp = (np.log(Tmat_C) + log_beta[t-1][:,:,None,None]).reshape((-1,S,A))
        log_beta[t] = (gam**(T-t))*lamb + softmax(temp)

    log_Z = softmax(log_alph[T].flatten())
    log_pt_lamb = np.zeros((T+1,S,A))
    for t in range(T):
        temp = (np.log(Tmat_C) + log_beta[T-t-1][:,:,None,None]).reshape((-1,S,A))
        log_pt_lamb[t] = log_alph[t] + log_beta[T-t-1]

    log_pt_lamb[T] = log_alph[T]

    log_pt_lamb -= log_Z

    pt_lamb = np.exp(log_pt_lamb)
    pt_lamb /= np.sum(pt_lamb,axis=(1,2))[:,None,None]
    
    return pt_lamb, log_Z

def get_policy(lamb,pt_emp,Tmat,p0,gam):
    S = Tmat.shape[0]
    A = Tmat.shape[1]
    T = pt_emp.shape[0] - 1
    log_alph = np.ones((T+1,S,A)) 
    log_beta = np.ones((T+1,S,A))

    
    #print(p0.shape,lamb.shape,log_alph[0].shape)

    log_alph[0] = np.log(p0)[:,None] + lamb
    log_beta[0] = (gam**T)*lamb

    Tmat_P = Tmat.reshape((-1,S))
    Tmat_C = Tmat.transpose(2,0,1)
    Tmat_C = Tmat_C[:,None,:,:]

    for t in range(1,T+1):
        log_alph[t] = (gam**t)*lamb + (softmax(np.log(Tmat_P) + log_alph[t-1].flatten()[:,None]))[:,None]
        temp = (np.log(Tmat_C) + log_beta[t-1][:,:,None,None]).reshape((-1,S,A))
        log_beta[t] = (gam**(T-t))*lamb + softmax(temp)

    log_pi = np.zeros((S,A))
    log_pi = log_beta[T] - softmax(log_beta[T].T)[:,None]
    
    return np.exp(log_pi)

def nll_func(lamb,pt_emp,Tmat,gam): #negative log likelihood
    S = Tmat.shape[0]
    A = Tmat.shape[1]
    lamb = lamb.reshape((S,A))
    
    T = pt_emp.shape[0] - 1
    p0 = np.sum(pt_emp[0],axis=-1)
    
    pt_lamb, log_Z = compute_pt_lamb(lamb,Tmat,p0,gam,T)
    
    emp_r = np.dot(gam**np.arange(T+1), np.sum(pt_emp*lamb[None,:,:],axis=(1,2)))
    print(-(emp_r - log_Z))
    return -(emp_r - log_Z)
    
def nll_grad(lamb,pt_emp,Tmat,gam): #negative log likelihood gradient
    S = Tmat.shape[0]
    A = Tmat.shape[1]
    lamb = lamb.reshape((S,A))
    
    T = pt_emp.shape[0] - 1
    p0 = np.sum(pt_emp[0],axis=-1)
    
    pt_lamb, log_Z = compute_pt_lamb(lamb,Tmat,p0,gam,T)
    
    #print(np.max(np.abs(np.sum(gam**np.arange(T+1)[:,None,None]*(pt_emp - pt_lamb),axis=0))))
    return -np.sum((gam**np.arange(T+1))[:,None,None]*(pt_emp - pt_lamb),axis=0).flatten()

def minimize_lamb(pt_emp,Tmat,gam,tol=1e-3): #optimize for the best fit reward (lamb) values
    S = Tmat.shape[0]
    A = Tmat.shape[1]
    
    lamb = 0.1*np.random.randn(S,A)  
    lamb = lamb.flatten()
    res = minimize(nll_func, x0 = lamb, method = 'L-BFGS-B', jac = nll_grad, args = (pt_emp,Tmat,gam),options={'disp':True},tol=tol)
    lamb = res.x.reshape((S,A))
    return lamb

def create_trajectories(states,actions,T):
    #create trajectories
    numtrajs = len(states) - (T + 1)
    trajs_s = np.zeros((numtrajs,(T+1)))
    trajs_a = np.zeros((numtrajs,(T+1)))
    for tr in range(numtrajs):
        trajs_s[tr] = states[tr:tr+T+1]
        trajs_a[tr] = actions[tr:tr+T+1]
    return trajs_s,trajs_a

def get_p0(pt_emp): #get probability of initial state by measuring frequency of each state. 
    return np.sum(pt_emp[0],axis=-1)

def get_empirical_frequencies(trajs_s,trajs_a,S,A): #get the probability of state-action pair s,a in each trajectory.
    #get empirical frequencies at each time step
    T = trajs_a.shape[1] - 1
    pt_emp = np.zeros((T+1,S,A))
    for t in range(T+1):
        for s in range(S):
            for a in range(A):
                pt_emp[t,s,a] = np.mean((trajs_s[:,t]==s)*(trajs_a[:,t]==a))    
    return pt_emp

def get_rewards(pt_emp,Tmat,gam,tol=1e-3):
	lamb = minimize_lamb(pt_emp,Tmat,gam,tol)
	return lamb

    
