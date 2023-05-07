#this is for model-based learning where the timing is remembered. 

from graph_navigator import GraphNavigator
import numpy as np
from copy import deepcopy
import networkx as nx
import sys

def create_binary_tree(levels):
    G = nx.Graph()
    G.add_nodes_from([('r',{'V':0, 'R':0})])
    for l in range(1,levels+1):
        for i in range(2**l):
            bin_i = bin(i)[2:].zfill(l)
            G.add_nodes_from([(bin_i+'r',{'V':0, 'R':0})])

    G.add_edge('r','0r')
    G.add_edge('r','1r')
    for i in list(G.nodes):
        for j in list(G.nodes):
            if len(i) == len(j)+1 and j == i[-len(j):]:
                G.add_edge(i,j)
    return G

class MazeSolver(GraphNavigator):
    def __init__(self, lambdas, prior_lambda, p0, rr_mean, tau = 500, p_thresh = 0.4, depth = 6, goal_state = '111111r', renewtime = 100, goal_reward = 1, end_nodes_explo_only = 0):
        self.depth = depth
        self.goal_state = goal_state

        self.G_timer = 0
        self.G_state = 1
        self.renewtime = renewtime

        self.p0 = p0
        self.lambdas = lambdas
        self.prior_lambda = prior_lambda
        self.rr_mean = rr_mean
        self.rr_noise = 0.0
        self.tau = tau
        self.p_thresh = p_thresh
        self.goal_reward = goal_reward
        self.end_nodes_explo_only = end_nodes_explo_only


    def initialize_navigator(self):
        G = create_binary_tree(self.depth)
        G.nodes[self.goal_state]['R'] = 1.0
        super().__init__(G)

    def initialize_graph(self,L):
        L.add_nodes_from([('r',{'V':0, 'R':0.05, 'p':self.p0, 'q_lambda': deepcopy(self.prior_lambda), 'rr': self.rr_mean,'T':self.tau, 'n':1})])
        return L

    def initialize_node(self):
        rr = self.rr_mean*(1 + self.rr_noise*np.random.randn())
        dict_els = {'V':0, 'R':0, 'p':self.p0, 'q_lambda': deepcopy(self.prior_lambda), 'rr':rr, 'T':self.tau, 'n':0}
        return dict_els

    def get_exp_reward(self,dict):
        p = dict['p']
        prob_lambda = dict['q_lambda']
        T = dict['T']
        rr = dict['rr']

        if p > self.p_thresh:
            return (1 - np.exp(-rr*T))
        else:
            return (1 - np.sum(np.exp(-self.lambdas*T)*prob_lambda)*np.exp(-rr*T))


    def update_node_attributes(self,state,obs): 
        for el in list(self.L.nodes):
            if el == state:
                T = self.L.nodes[el]['T']
                rr = self.L.nodes[el]['rr']
                p = self.L.nodes[el]['p']
                temp = np.exp(-rr*T)
                
                if obs == 1: #if reward is received
                    integ = np.sum(np.exp(-self.lambdas*self.L.nodes[el]['T'])*self.L.nodes[el]['q_lambda'])
                    self.L.nodes[el]['p'] *= (1 - temp)/((1 - temp)*p + (1-p)*(1-integ))
                    self.L.nodes[el]['q_lambda'] *= (1 - np.exp(-self.lambdas*self.L.nodes[el]['T']))
                    self.L.nodes[el]['q_lambda'] /= np.sum(self.L.nodes[el]['q_lambda'])
                else: #if reward is not received
                    integ = np.sum(np.exp(-self.lambdas*self.L.nodes[el]['T'])*self.L.nodes[el]['q_lambda'])
                    self.L.nodes[el]['p'] *= temp/(p*temp + (1-p)*integ)
                    self.L.nodes[el]['q_lambda'] *= np.exp(-self.lambdas*self.L.nodes[el]['T'])
                    self.L.nodes[el]['q_lambda'] /= np.sum(self.L.nodes[el]['q_lambda'])
                
                self.L.nodes[el]['T'] = 0
                self.L.nodes[el]['n'] += 1
            else:
                self.L.nodes[el]['T'] += 1 + 0.25*np.random.randn()
                self.L.nodes[el]['T'] = max(0,self.L.nodes[el]['T'])
            
            if len(list(self.L[el])) > 3*(1-self.end_nodes_explo_only) + 1*self.end_nodes_explo_only: #gives zero reward to non-leaf nodes if end_nodes_explo_only = 1.
                self.L.nodes[el]['R'] = 0
            else:
                self.L.nodes[el]['R'] = self.get_exp_reward(self.L.nodes[el])


    def update_environment_graph(self,state,rew,obs):
        if obs == 1:
            self.G_state = 0
            self.G_timer = 0
        
        if self.G_state == 0:
            if self.G_timer < self.renewtime:
                self.G_timer += 1
            else:
                self.G_state = 1
                self.G_timer = 0
        else:
            self.G_timer = 0
            
    def get_reward_obs(self,state):
        reward = 0
        observation = 0
        
        if self.G_state == 1 and state == self.goal_state:
            reward = self.goal_reward
            if reward > 1e-5:
                observation = 1
            
        return reward,observation