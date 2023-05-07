import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import networkx as nx
import sys
import matplotlib
from matplotlib import cm

from MM_Plot_Utils import plot, hist
from MM_Maze_Utils import *
from MM_Traj_Utils import *


#general utils

def sig(x):
    return 1/(1 + np.exp(-x))

def convert_node_to_pos(node):
    d = len(node)-1
    pos = np.array([0.0,0.0])
    th = 0
    for i in range(d):
        a = node[i+1]
        if a == '0':
            th += 90*np.pi/180
        else:
            th -= 90*np.pi/180
        v = np.array([np.cos(th),np.sin(th)])
        t = 1.0/2**(np.floor(i/2)+1)
        pos += v*t
    return np.array(pos)

def get_data_index_to_node_dict(depth):
    node_list = [['r']]
    for i in range(depth):
        temp_list = []
        for node in node_list[i]:
            pos = convert_node_to_pos(node)

            node_L = node + '0'
            node_R = node + '1'
            pos_L = convert_node_to_pos(node_L)
            pos_R = convert_node_to_pos(node_R)
            if 1e-5 + pos_L[0] < pos_R[0] or pos_L[1] > pos_R[1]+1e-5:
                temp_list += [node_L]
                temp_list += [node_R]
            else:
                temp_list += [node_R]
                temp_list += [node_L] 

        node_list += [temp_list]

    node_list_ordered = []
    for s in node_list:
        node_list_ordered += s
        
    node_list_ordered += ['h']

    return node_list_ordered

def append_reward_to_ptn(ptn,tf,rew=True):
    if rew == False: #unrewarded animals
        new_col = np.ones((len(ptn),1),dtype=int)
        ptn_rew = np.concatenate((ptn,new_col),1)
        return ptn_rew
    #get the times at which the animal leaves the water port
    leave_times = []
    for ip,p in enumerate(ptn):
        frnum = tf.no[p[0]][p[1]+p[2]+1,1] + tf.fr[p[0],0]
        a = TimeInMaze(frnum,tf)
        leave_times += [a]
    leave_times = np.array(leave_times)

    #get the times when reward is received
    reward_times = []
    for i in range(len(tf.re)):
        frnums = tf.re[i][:,0] + tf.fr[i,0]
        r_time = [TimeInMaze(frnum,tf) for frnum in frnums]
        reward_times += r_time

    new_col = np.zeros((len(ptn),1),dtype=int)
    ptn_rew = np.concatenate((ptn,new_col),1)

    #reward is obtained if the animal leaves the water port after the reward is received
    for r in reward_times:
        filt = leave_times > r
        index = np.arange(len(leave_times))[filt][0] 
        ptn_rew[index,4] = 1
    return ptn_rew

def convert_runs_to_mazenodes_for_plot(seqs):
    depth = 6
    data_to_node = get_data_index_to_node_dict(depth)
    paths = []
    for seq in seqs:
        sample = [data_to_node.index(s) for s in seq]
        paths += [sample]
    return paths

def get_nodes_from_data(nickname,data_to_node, thresh = 10):
    tf = LoadTraj(nickname+'-tf')
    nodes = []
    times = []
    for i in range(len(tf.no)):
        if tf.no[i].shape[0] > 2*thresh:
            indices = tf.no[i][thresh:-thresh-1,0] #remove run from home and to home (run here defined as n=thresh decisions).
        else:
            continue
            
        ts = tf.fr[i][0] + tf.no[i][:,1]
        nodes_in_bout = [data_to_node[i] for i in indices]
        nodes += nodes_in_bout
        times += list(ts/30) #30Hz framerate
    return nodes,times

def get_nodes_from_data_episodic(nickname,data_to_node, thresh = 2):
    tf = LoadTraj(nickname+'-tf')
    nodes = []
    times = []
    for i in range(len(tf.no)):
        if tf.no[i].shape[0] > 2*thresh:
            indices = tf.no[i][thresh:-thresh-1,0] #remove run from home and to home (run here defined as n=thresh decisions).
        else:
            continue
            
        ts = tf.fr[i][0] + tf.no[i][:,1]
        nodes_in_bout = [data_to_node[i] for i in indices]
        nodes += [nodes_in_bout]
        times += [list(ts/30)] #30Hz framerate
    return nodes,times

def get_actions_from_nodes(nodes,depth):
    #Get actions from nodes
    actions = []
    forced = []
    #actions: 0 down-to-left, 1 down-to-right, 2-up-from-left, 3-up-from-right, 4-stay at same place
    #forced: 1 if going down from root or going up from leaf. 0 otherwise.

    for i in range(len(nodes)-1):
        if len(nodes[i+1]) > len(nodes[i]):
            actions += [int(nodes[i+1][-1])]    
            if len(nodes[i]) == 1:
                forced += [1]
            else:
                forced += [0]
        elif len(nodes[i+1]) < len(nodes[i]):
            actions += [int(nodes[i][-1])+2]
            if len(nodes[i]) == depth + 1:
                forced += [1]
            else:
                forced += [0]
        if nodes[i] == nodes[i+1]:
            actions += [4]
            forced += [0]
            
    return actions,forced


def add_diedges_to_graph(G,edges):
    index = 0
    for e in edges:
        G,index = add_diedge(G,e[0],e[1],index)
    return G

def add_diedge(G,node1,node2,index):
    G.add_edges_from([(node1,node2,{'Qe':0, 'Qr':0, 'V':0, 'R':0, 'i':index}),(node2,node1,{'Qe':0, 'Qr':0, 'V':0,'R':0, 'i':index+1})])
    index += 2
    return G,index

def create_binary_tree(levels):
    G = nx.Graph()
    G.add_nodes_from([('r',{'V':0, 'R':0})])
    for l in range(1,levels+1):
        for i in range(2**l):
            bin_i = bin(i)[2:].zfill(l)
            G.add_nodes_from([('r'+ bin_i[::-1],{'V':0, 'R':0})])

    G.add_edge('r','r0')
    G.add_edge('r','r1')
    for i in list(G.nodes):
        for j in list(G.nodes):
            if len(i) == len(j)+1 and j == i[:len(j)]:
                G.add_edge(i,j)
    return G

def create_binary_tree_directed(levels):
    G = nx.DiGraph()
    G.add_nodes_from([('r')])
    #index = 0
    for l in range(1,levels+1):
        for i in range(2**l):
            bin_i = bin(i)[2:].zfill(l)
            G.add_nodes_from([('r'+ bin_i[::-1])])
    edges = []
                    
    edges += [('r','r0')]
    edges += [('r','r1')]
    
    for i in list(G.nodes):
        for j in list(G.nodes):
            if len(i) == len(j)+1 and j == i[:len(j)] and (i,j) not in edges and (j,i) not in edges:
                edges += [(i,j)]
    G = add_diedges_to_graph(G,edges)
    return G

def get_state_index(state,G): #converts the node-node pair of a directed edge on G to its index. 
    if (state[0],state[1]) in list(G.edges):
        return G[state[0]][state[1]]['i']
    else:
        return -1
    
def get_index_to_state_vec(G): #converts the index of an directed edge to its node-node pair in G. 
    index_to_state = []
    for i in range(len(list(G.edges))):
        index_to_state += [0]
    for e in list(G.edges):
        e1 = e[0]
        e2 = e[1]
        index_to_state[G[e1][e2]['i']] = e
    return index_to_state

def find_distance_bn_nodes(node1,node2):
    minl = min(len(node1),len(node2))
    maxl = max(len(node1),len(node2))
    counter = 0
    for i in range(minl):
        if node1[i] == node2[i]:
            counter += 1
        else:
            break
    return (maxl - counter) + (minl - counter)

#quantifying exploration statistics

def get_four_biases(actions,forced):#Get biases
    downs_from_stem = 0
    ups_from_stem = 0
    alts_from_stem = 0

    up_from_branch = 0
    reverses_from_branch = 0
    down_from_branch = 0

    for i in range(len(actions)-1):
        #From stem
        if forced[i+1] != 1 and actions[i+1] != 4: #condition to not count going into leaf
            if actions[i] == 0 or actions[i] == 1 and forced[i+1] != 1: 
                if actions[i+1] == 0 or actions[i+1] == 1:
                    downs_from_stem += 1
                elif actions[i+1] == 2 or actions[i+1] == 3:
                    ups_from_stem += 1

            if (actions[i] == 0 and actions[i+1] == 1) or (actions[i] == 1 and actions[i+1] == 0):
                alts_from_stem += 1

        #From branch
        if forced[i+1] != 1 and actions[i+1] != 4: #condition to not count going into root
            if (actions[i] == 2 or actions[i] == 3) and (actions[i+1] == 2 or actions[i+1] == 3):
                up_from_branch += 1

            if (actions[i] == 2 and actions[i+1] == 0) or (actions[i] == 3 and actions[i+1] == 1):
                reverses_from_branch += 1

            if (actions[i] == 2 and actions[i+1] == 1) or (actions[i] == 3 and actions[i+1] == 0):
                down_from_branch += 1


    P_SF = downs_from_stem/(ups_from_stem+downs_from_stem)  
    P_SA = alts_from_stem/downs_from_stem
    P_BF = (up_from_branch + down_from_branch)/(up_from_branch + down_from_branch + reverses_from_branch)
    P_BS = up_from_branch/(up_from_branch + down_from_branch)
    
    return [P_SF,P_SA,P_BF,P_BS]

#utils for runs and paths to goal
        
def get_run_lengths_to_goal(nodes,goal_nodes,passing_nodes = None):
    run_lengths = []
    times = []
    for i,s in enumerate(nodes):
        if s in goal_nodes:
            temp = 0
            while i-temp > 0 and find_distance_bn_nodes(s,nodes[i-temp-1]) == temp+1 and find_distance_bn_nodes(nodes[i-temp],nodes[i-temp-1])==1:
                temp += 1
            if passing_nodes != None and passing_nodes not in nodes[i-temp:i]:
                continue
            times += [i]
            run_lengths += [temp]
            
            #print(s)
    return np.array(run_lengths), np.array(times)

def process_run_lengths(runl,times,thresh_run):
    cuml = np.zeros(len(runl))
    for i in range(len(runl)):
        cuml[i] = np.sum(runl[:i] > thresh_run)
    filt = [i for i in range(len(cuml)-1) if cuml[i] != cuml[i+1]]
    return cuml[filt],times[filt]

def get_runs_to_goal(nodes,goal,thresh = 0):
    seqs = []
    for i,s in enumerate(nodes):
        if s == goal:
            temp = 0
            seq = [s]
            while i-temp > 0 and find_distance_bn_nodes(s,nodes[i-temp-1]) == temp+1 and find_distance_bn_nodes(nodes[i-temp],nodes[i-temp-1])==1:
                seq += [nodes[i-temp-1]]
                temp += 1
            seq = seq[::-1]
            if len(seq) > thresh + 1:
                seqs += [seq]
    return seqs

def get_nodes_from_start_to_goal(start,goal):
    nodes = [start]
    d = 0
    while start[:d+1] == goal[:d+1]:
        d += 1

    for i in range(len(start) - d):
        nodes += [start[:-i-1]]
    for i in range(len(goal) - d):
        nodes += [goal[:d+i+1]]
        
    return nodes

def get_states_from_start_to_goal(start,goal):
    nodes = get_nodes_from_start_to_goal(start,goal)
    path = []
    for i in range(len(nodes)-1):
        path += [(nodes[i],nodes[i+1])]
        
    return path[::-1]

def get_direct_paths(time_rews,L):
    time_paths = [0]
    cum_paths = [0]
    for i in range(len(time_rews)-1):
        dt = time_rews[i+1] -  time_rews[i]
        if dt == L-1:
            time_paths += [time_rews[i+1]]
            cum_paths += [cum_paths[-1] + 1]
    return time_paths, cum_paths


def plot_graph_Qr(G, save_name = None, episode = None, starts = None, goals = None, vmax = 5, style= None,\
                 figsize=(6.5,6), show = None,arrowsize = 12):
    plt.close("all")
    fig,axis = plt.subplots(1,1,figsize = figsize)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax, clip=True)
    cmap = cm.jet
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    cs_e =[]
    for l in list(G.edges):
        v = np.max(G[l[0]][l[1]]['Qr'])
        if v < 1e-3:
            cs_e += [(0,0,0,0.1)]
        else:
            cs_e += [mapper.to_rgba(v)]
            
    if starts is not None:
        for start in starts:
            for k,n in enumerate(list(G.edges)):
                if n[0] == start[0] and n[1] == start[1]:
                    cs_e[k] = matplotlib.colors.to_rgba("C2")
    if goals is not None:
        for goal in goals:
            for k,n in enumerate(list(G.edges)):
                if n[0] == goal[0] and n[1] == goal[1]:
                    cs_e[k] =  matplotlib.colors.to_rgba("C3")

    if show is not None:
        for s in show:
            for k,n in enumerate(list(G.edges)):
                if n[0] == s[0] and n[1] == s[1]:
                    cs_e[k] = matplotlib.colors.to_rgba("C8")    

    node_size = []
    for k,n in enumerate(list(G.nodes)):
        if isinstance(n,str):
            if ':' in n:
                node_size += [5]
            else:
                node_size += [20]
        else:
            node_size += [20]

    

    if style is None:
        im = nx.draw_kamada_kawai(G, connectionstyle='arc3, rad = 0.2', ax = axis, edge_color = cs_e, arrowsize = arrowsize, node_size = node_size, node_color = 'k', alpha = 0.75)
    elif style == 'planar':
        im = nx.draw_planar(G, connectionstyle='arc3, rad = 0.2', ax = axis, edge_color = cs_e, arrowsize = arrowsize, node_size = node_size, node_color = 'k', alpha = 0.75)
    elif style == 'spectral':
        im = nx.draw_spectral(G, connectionstyle='arc3, rad = 0.2', ax = axis, edge_color = cs_e, arrowsize = arrowsize, node_size = node_size, node_color = 'k', alpha = 0.75)
    
    if episode is not None:
        axis.text(1,1,"Episode: "  + "{0:4d}".format(episode), fontsize = 16, fontfamily = "sans-serif")
    mapper._A = []
    cb  = fig.colorbar(mapper, shrink = 0.4)
    cb.set_label(r"$\max_a q_r(s,a)$", fontsize = 15, labelpad = 10)
    cb.ax.tick_params(labelsize = 15)
    
    if save_name is not None:
        fig.savefig(save_name, dpi = 100)
    else:
        plt.show()

def create_pngs_for_movie(Gs,starts,goals,time_rews,cum_rews, L, rew, name_prefix, style = None, figsize = (6.5,6), show = None,arrowsize = 12):
    time_paths, cum_paths = get_direct_paths(time_rews,L)
    cum_paths_by_epi = [0]
    for i in range(1,len(time_rews)):
        if time_rews[i] in time_paths:
            cum_paths_by_epi += [cum_paths_by_epi[-1] + 1]
        else:
            cum_paths_by_epi += [cum_paths_by_epi[-1]]

    for i,G in enumerate(Gs[:]):
        save_name = "./movies/" + name_prefix + "_graph_%04d.png"%(i)
        plot_graph_Qr(Gs[i],save_name = save_name, starts = starts, goals = goals, episode = i+1, vmax = rew, figsize = figsize,style = style, show = show,arrowsize = arrowsize)

        plt.close("all")
        fig,ax1 = plt.subplots(1,1,figsize = figsize)
        ax1.tick_params(labelsize = 14)
        ax = ax1.twinx()
        ax.plot(time_rews[1:i], cum_rews[1:i],'g-')
        ax.tick_params(labelsize = 14)
        ax1.plot(time_rews[1:i], cum_paths_by_epi[1:i],'r-')
        ax1.set_xlim(0,np.max(time_rews)*1.1)
        ax1.set_ylim(-2,cum_paths[-1]*1.1)
        ax.set_ylim(0,len(cum_rews)*1.1)
        ax1.set_ylabel(r"$\#$ of direct paths from start", fontsize= 17, color = 'C3')
        ax.set_ylabel(r"$\#$ of rewards", fontsize= 17, color = 'C2')
        ax1.set_xlabel("Time (steps)", fontsize= 17)
        for sp in ['top','bottom','left','right']:
            ax1.spines[sp].set_linewidth(1.25)

        ax1.text(np.max(time_rews)*1.6,cum_paths[-1]*1.6," ", fontsize = 18, fontfamily = "sans-serif")
        ax1.text(-np.max(time_rews)*0.4,-cum_paths[-1]*0.4," ", fontsize = 18, fontfamily = "sans-serif")

        fig.tight_layout()
        fig.savefig("./movies/" + name_prefix + "_curve_%04d.png"%(i), dpi = 100)

#optimization of L_ramp, L_step and L_sig

from scipy.optimize import minimize
def r_ramp(a,b,c,t):
    return (a + b*t + c*(t**2))

def r_step(rb,ra,ts,t):
    return rb*(t < ts) + ra*(t >= ts)

def r_sig(ri, delr, tau, beta ,t):
    return ri + delr*sig(beta*(t-tau))

def nL_sig(x,times):
    times= np.array(times)
    T = times[-1]
    times = times/T


    ri = x[0]
    delr = x[1]
    tau = x[2]
    beta = x[3]
    
    T = times[-1]
    nL = -np.sum(np.log(r_sig(ri, delr, tau, beta,times) + 1e-10)) \
           + (ri*T + (delr/beta)*np.log(sig(beta*tau)/sig(-beta*(T-tau))))
    #print(x,T,-nL)
    return nL

def dnL_sig(x,times):

    times= np.array(times)
    T = times[-1]
    times = times/T

    ri = x[0]
    delr = x[1]
    tau = x[2]
    beta = x[3]
    
    T = times[-1]
    
    sigs = sig(beta*(times-tau))
    r_sigs = r_sig(ri, delr, tau, beta,times)
    
    dLs = np.zeros(4)
    
    dLs[0] = np.sum(1/r_sigs) - T
    
    dLs[1] = np.sum(sigs/r_sigs) \
    - (1/beta)*np.log(sig(beta*tau)/sig(-beta*(T-tau)))
    
    dLs[2] = np.sum(-beta*delr*sigs*(1-sigs)/r_sigs) \
    - delr*(sig(-beta*tau) - sig(beta*(T-tau)))
    
    dLs[3] = np.sum((times-tau)*delr*sigs*(1-sigs)/r_sigs) \
    - (delr/beta)*(-(1/beta)*np.log(sig(beta*tau)/sig(-beta*(T-tau))) + (T-tau)*sig(beta*(T-tau)) + tau*sig(-beta*tau))
    
    return -dLs

def L_sig_opt(times):
    times= np.array(times)
    T = times[-1]
    times = times/T

    x0 = [1e-3,1e-1,0.5, 1e-2]
    bounds = [[1e-10,np.inf],[0,np.inf],[0,1],[1e-10,50]]
    res = minimize(nL_sig,x0,bounds = bounds, jac = dnL_sig, args = (times))
    #a,b,c = res.x

    return res.x

def nL_ramp(x,times):
    times= np.array(times)
    T = times[-1]
    times = times/T


    a= x[0]
    b = x[1]
    c = x[2]
    
    T = times[-1]
    nL = -(np.sum(np.log(r_ramp(a,b,c,times) + 1e-10)) - (a*T + b*(T**2/2) + c*(T**3/3)))
    #print(x,T,-nL)
    return nL

def dnL_ramp(x,times):

    times= np.array(times)
    T = times[-1]
    times = times/T

    a= x[0]
    b = x[1]
    c = x[2]
    
    T = times[-1]
    
    dLs = np.zeros(3)
    
    dLs[0] = np.sum(1/r_ramp(a,b,c,times)) - T
    dLs[1] = np.sum(times/r_ramp(a,b,c,times)) - T**2/2
    dLs[2] = np.sum(times**2/r_ramp(a,b,c,times)) - T**3/3
    
    return -dLs

def L_ramp_opt(times):
    times= np.array(times)
    T = times[-1]
    times = times/T

    x0 = [1e-4,1e-3,1e-5]
    bounds = [[1e-10,np.inf],[0,np.inf],[0,np.inf]]
    res = minimize(nL_ramp,x0,bounds = bounds, jac = dnL_ramp, args = (times))
    a,b,c= res.x

    return res.x

def L_step(x,times):
    times= np.array(times)
    T = times[-1]
    times = times/T

    ts = x[0]
    m = np.sum(times < ts)
    n = len(times)
    T = times[-1]
    L = (m*np.log(m/ts) + (n-m)*np.log((n-m)/(T - ts)) - n)
    return L

def L_step_opt(times):
    times= np.array(times)
    T = times[-1]
    times = times/T

    Lsteps = np.zeros(len(times)-2)
    n = len(times)
    T = times[-1]
    for i,ts in enumerate(times[1:-1]):
        m = np.sum(times < ts)
        Lsteps[i] = (m*np.log(m/ts) + (n-m)*np.log((n-m)/(T - ts)) - n)
        #print(Lsteps[i],m,ts,T)
        
    ind = np.argmax(Lsteps)
    ts = times[1:-1][ind]
    
    rb = np.sum(times<ts)/ts
    ra = np.sum(times>ts)/(T - ts)
    
    return ts,rb,ra

def get_Lstep_Lramp_Lsig(nodes,goal_node,thresh_run,plot = False,xmax = 10000,ymax = 60):
    runl_goal,times_goal = get_run_lengths_to_goal(nodes,[goal_node])
    cuml_lr,timl_lr = process_run_lengths(runl_goal,times_goal,thresh_run)

    x_ramp = L_ramp_opt(timl_lr)
    Lramp = -nL_ramp(x_ramp,timl_lr)

    T = timl_lr[-1]
    p = np.polyfit(timl_lr, cuml_lr/timl_lr,2) #unused

    x_step = L_step_opt(timl_lr)
    Lstep = L_step([x_step[0]],timl_lr)

    x_sig = L_sig_opt(timl_lr)
    Lsig = -nL_sig(x_sig,timl_lr)



    # print(x_ramp)
    # print(x_step)
    # print(x_sig)

    if plot:
        plt.close("all")
        fig,axis = plt.subplots(1,1,figsize = (4.5,4))
        t = np.linspace(0,timl_lr[-1],200)
        a,b,c = x_ramp

        a/=T
        b/=T**2
        c/=T**3
        ramp_pred = a*(t) + b*(t**2/2) + c*(t**3/3)
        ts,rb,ra = x_step
        ts *= T
        rb /= T
        ra /= T
        step_pred = rb*t*(t<ts) + (t>ts)*(rb*ts + ra*(t-ts))
        
        ri,delr,tau,beta = x_sig
        tau *= T
        beta /= T
        ri /= T
        delr /= T
        sig_pred = ri*t + (delr/beta)*np.log(sig(beta*tau)/sig(-beta*(t-tau)))
        print(ri,ri + delr, 1/beta, tau)

        axis.plot(timl_lr,cuml_lr,'r-',lw=3)

        axis.plot(t,ramp_pred,'C0-',lw=2)
        axis.plot(t,step_pred,'C1-',lw=2)
        axis.plot(t,sig_pred,'C2-',lw=2)
    return Lstep,Lramp, Lsig

def fit_sigmoid_directpath_rate(nodes,goal_node,thresh_run):
    runl_goal,times_goal = get_run_lengths_to_goal(nodes,[goal_node])
    cuml_lr,timl_lr = process_run_lengths(runl_goal,times_goal,thresh_run)


    T = timl_lr[-1]

    x_sig = L_sig_opt(timl_lr)

    ri,delr,tau,beta = x_sig
    tau *= T
    beta /= T
    ri /= T
    delr /= T

    return ri,delr,tau,beta


