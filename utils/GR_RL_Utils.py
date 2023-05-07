import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import networkx as nx
from GR_Maze_Utils import *


def get_action(state,G): #this is softmax policy and correct
    Qs = G[state[0]][state[1]]['Qe'] + G[state[0]][state[1]]['Qr']
    probs = np.exp(Qs)/np.sum(np.exp(Qs))
    return np.random.choice(len(probs),p=probs)

def get_explo_action(state,G): #this is softmax policy with exploration parameters only
    Qs = G[state[0]][state[1]]['Qe']
    probs = np.exp(Qs)/np.sum(np.exp(Qs))
    return np.random.choice(len(probs),p=probs)

def get_next_state(s,a,Tmat): #OK
    probs = Tmat[s,a]/np.sum(Tmat[s,a])
    return np.random.choice(len(probs),p=probs)

def get_Q_from_G(G,state,action): #OK, get Q(s,a), maybe not necessary
    return G[state[0]][state[1]]['Qr'][action] + G[state[0]][state[1]]['Qe'][action]

def get_delta(sasr,goal,gam,G, mode = 0):
    prevstate = sasr[0]
    prevaction = sasr[1]
    state = sasr[2]
    rew = sasr[3]
    
    Qs = G[state[0]][state[1]]['Qr']
    Qps = G[prevstate[0]][prevstate[1]]['Qr'][prevaction]
    
    if state == goal:
        delta = rew - Qps
    else:
        if mode == 0:
            delta = rew + gam*np.max(Qs) - Qps
        else:
            Qs = G[state[0]][state[1]]['Qe'] + G[state[0]][state[1]]['Qr']
            probs = np.exp(Qs)/np.sum(np.exp(Qs))
            Qexp = np.sum(G[state[0]][state[1]]['Qr']*probs)
            nextaction = get_action(state,G)
            #delta = rew + gam*Qs[nextaction] - Qps
            delta = rew + gam*Qexp - Qps #expected SARSA
        
    return delta

def update_memory_q(q_mem, sasr, q_mem_size):
    if len(q_mem) >= q_mem_size:
        new_mem = deepcopy(q_mem[1:] + [sasr[:2]])
    else:
        new_mem = deepcopy(q_mem + [sasr[:2]])
        
    return new_mem

def update_memory_dyna(dyna_mem, sasr_seq,dyna_mem_size): 
    new_mem = deepcopy(dyna_mem + sasr_seq)
    if len(new_mem) >= dyna_mem_size:
        new_mem = new_mem[-dyna_mem_size:]
    return new_mem

def update_Q_Q(q_mem,sasr,goal,G,params):#Q(lambda)
    alpha = params[0]
    gam = params[1]
    lamb = params[2] + 1e-6
    
    delta = get_delta(sasr,goal,gam,G, mode=0)
    
    for i in range(len(q_mem)):
        prevstate = q_mem[-1-i][0]
        prevaction = q_mem[-1-i][1]
        nextstate = q_mem[-i][0]
        nextaction = q_mem[-i][1]
        if i > 0 and prevstate[0] == nextstate[1]:
            break
        G[prevstate[0]][prevstate[1]]['Qr'][prevaction] += alpha*delta*(lamb**i)
    return G

def update_Q_sarsa(q_mem,sasr,goal,G,params):#expected sarsa(lambda)
    alpha = params[0]
    gam = params[1]
    lamb = params[2] + 1e-6
    
    delta = get_delta(sasr,goal,gam,G, mode= 1)
    
    for i in range(len(q_mem)):
        prevstate = q_mem[-1-i][0]
        prevaction = q_mem[-1-i][1]
        nextstate = q_mem[-i][0]
        nextaction = q_mem[-i][1]
        if i > 0 and prevstate[0] == nextstate[1]:
            break
        G[prevstate[0]][prevstate[1]]['Qr'][prevaction] += alpha*delta*(lamb**i)
    return G

def update_Q_sQ(q_mem,sasr,goal,G,params): #sQ(lambda)
    alpha = params[0]
    gam = params[1]
    lamb = params[2] + 1e-6
    
    state = sasr[2]
    if state != goal:
        return G
    
    delta = get_delta(sasr,goal,gam,G)
    
    for i in range(len(q_mem)):
        prevstate = q_mem[-1-i][0]
        prevaction = q_mem[-1-i][1]
        nextstate = q_mem[-i][0]
        nextaction = q_mem[-i][1]
        if i > 0 and prevstate[0] == nextstate[1]:
            break
        if i == 0:
            G[prevstate[0]][prevstate[1]]['Qr'][prevaction] += alpha*delta
        elif i > 0:
            Qs_next = get_Q_from_G(G,nextstate,nextaction)
            Qps_prev =  get_Q_from_G(G,prevstate,prevaction)
            
            Qs_next = G[nextstate[0]][nextstate[1]]['Qr'][nextaction]
            Qps_prev = G[prevstate[0]][prevstate[1]]['Qr'][prevaction]

            temp1 = G[prevstate[0]][prevstate[1]]['Qr'][prevaction]
            G[prevstate[0]][prevstate[1]]['Qr'][prevaction] += alpha*(lamb**i)*(gam*Qs_next - Qps_prev)
            temp2 = G[prevstate[0]][prevstate[1]]['Qr'][prevaction]
    return G   

def update_Q_dyna(dyna_mem,sasr,goal,G,params): #Dyna
    alpha = params[0]
    gam = params[1]
    lamb = params[2] + 1e-6
    dyna_plan_steps = int(params[3])
    
    delta = get_delta(sasr,goal,gam,G)
    
    #RL
    prevstate = sasr[0]
    prevaction = sasr[1]
    G[prevstate[0]][prevstate[1]]['Qr'][prevaction] += alpha*delta
    
    #planning
    if len(dyna_mem) > dyna_plan_steps:
        for i in range(dyna_plan_steps):
            k = np.random.choice(len(dyna_mem))
            sasr = dyna_mem[k]
            delta = get_delta(sasr,goal,gam,G)
            prevstate = sasr[0]
            prevaction = sasr[1]
            G[prevstate[0]][prevstate[1]]['Qr'][prevaction] += alpha*delta
        
    return G

def run_RL_simulation(G,Tmat,starts,goal,nepi,niter,renewtime, alpha = 0.01, gam = 0.98,lamb = 0.8,rew = 20, mode = 0,\
                      renew = False, q_mem_size = 15, dyna_mem_size = 500,dyna_plan_steps = 10):

    np.set_printoptions(precision = 3, suppress=True)
    index_to_state = get_index_to_state_vec(G)
    
    rew_timer = 0
    G[goal[0]][goal[1]]['R'] = rew

    nodes = []

    rews_received = 0
    cum_rews = []
    time_rews = []
    last_time = 0
    time_visits = []
    clock = 0
    
    params = [alpha,gam,lamb]
    
    dyna_mem = []
    
    Gs = []
    for epi in range(nepi):
        #initialize state
        state = deepcopy(starts[np.random.choice(len(starts))])
        nodes += [state[0]]
        nodes += [state[1]]

        q_mem = []
        for i in range(niter):
            rew_timer += 1
            clock += 1
            last_time += 1
            s_index = get_state_index(state,G)
            a = get_action(state,G)
            
            s_next = get_next_state(s_index,a,Tmat)
            prevstate = deepcopy(state)
            state = index_to_state[s_next]
            Qs = G[prevstate[0]][prevstate[1]]['Qe'] + G[prevstate[0]][prevstate[1]]['Qr']
            probs = np.exp(Qs)/np.sum(np.exp(Qs))
            #print(i,prevstate[0],prevstate[1],state[1], probs)
            curr_rew = G[state[0]][state[1]]['R']

            nodes += [state[1]]
            
            sasr = [prevstate, a, state, curr_rew]
            
            if mode == 3:
                dyna_mem = update_memory_dyna(dyna_mem, sasr, dyna_mem_size)
            else:
                q_mem = update_memory_q(q_mem, sasr, q_mem_size)
            
            if mode == 0:
                G = update_Q_sQ(q_mem,sasr,goal,G,params) #sQ
            elif mode == 1:
                G = update_Q_Q(q_mem,sasr,goal,G,params) #Q
            elif mode == 2:
                G = update_Q_sarsa(q_mem,sasr,goal,G,params) #sarsa
            elif mode == 3:
                G = update_Q_dyna(dyna_mem,sasr,goal,G,params)

            if state == goal:
                if renew:
                    if G[goal[0]][goal[1]]['R'] > 1e-6:
                        rews_received += 1
                        cum_rews += [rews_received]
                        time_rews += [clock]
                        rew_timer = 0 
                    time_visits += [last_time]
                    last_time = 0
                else:
                    rews_received += 1
                    cum_rews += [rews_received]
                    time_rews += [rew_timer] 
                    time_visits += [last_time]
                    last_time = 0
                    break
                    
            if renew:
                if rew_timer > renewtime:
                    G[goal[0]][goal[1]]['R'] = rew
                else:
                    G[goal[0]][goal[1]]['R'] = 0
                    
        Gstore = deepcopy(G)
        Gs += [Gstore]
        
                    
            #print(state[0],a,state[-1],rew,last_time)

    return nodes,time_rews,cum_rews,Gs



