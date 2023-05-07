import numpy as np
from copy import deepcopy
import networkx as nx
from abc import ABC, abstractmethod


class GraphNavigator(ABC):
    def __init__(self, G, gam = 0.9, niter_V = 10, policytype=0 , eps_greedy = 1.0, t_softmax = 0.3):
        self.G = G #environment graph defined as an nx graph with attribute 'R' storing the rewards
        L = nx.Graph()
        self.L = deepcopy(L)

        self.L = self.initialize_graph(self.L)

        self.gam = gam
        self.niter_V = niter_V
        self.eps_greedy = eps_greedy
        self.t_softmax = t_softmax
        self.policytype = policytype

    @abstractmethod
    def initialize_graph(self,L):
        pass

    @abstractmethod
    def initialize_node(self):
        pass

    def expand_graph(self,state):
        G_nbrs = list(self.G[state])
        for el in G_nbrs:
            if el not in list(self.L.nodes):
                dict_els = self.initialize_node()
                self.L.add_nodes_from([(el,dict_els)])
                self.L.add_edge(state,el)

    @abstractmethod
    def update_node_attributes(self,state,obs):
        pass

    def get_policy(self,vals): #polictype = 0 is softmax, =1 is eps_greedy 
        policy = np.zeros(len(vals))
        policytype = self.policytype
        
        if policytype == 1:
            policy += self.eps_greedy/len(vals)
            policy[np.argmax(vals)] += 1 - self.eps_greedy

        else:
            maxval = np.max(vals)
            minval = np.min(vals)

            temp = self.t_softmax #temperature
            
            if len(vals) > 1 and maxval != minval:
                for i in range(len(vals)):
                    policy[i] = np.exp((1.0/temp)*(vals[i]-minval)/(maxval-minval))
            else:
                policy = np.ones(len(vals))/len(vals)
                
            policy /= np.sum(policy)

        return policy

    def evaluate_V(self): #use value iteration to compute the value function
        niter = self.niter_V
        for i in range(niter):
            for el in list(self.L.nodes):
                vals = []
                nghbrs = list(self.L[el]) + [el]
                for el_nghbrs in nghbrs:
                    vals += [self.L.nodes[el_nghbrs]['V']]
                vals = np.array(vals)
                self.L.nodes[el]['V'] = self.L.nodes[el]['R'] + self.gam*np.dot(vals,self.get_policy(vals))

    def get_next_state(self,state): #execute policy
        vals = []
        nghbrs = list(self.L[state]) + [state]
        for nghbr in nghbrs:
            vals += [self.L.nodes[nghbr]['V']]
        vals = np.array(vals)
        #print(vals)
        policy = self.get_policy(vals)
        return nghbrs[np.random.choice(len(policy),p=policy)]

    @abstractmethod
    def update_environment_graph(self,state,rew,obs): #Updates the state of the real environment. 
        pass

    @abstractmethod
    def get_reward_obs(self,state): #Get rewards and observations
        pass

    def run(self,init_state, niter):
        state = deepcopy(init_state)

        states = [state]
        rewards = [0]
        observations = [0]
        for i in range(niter):
            if i%250 == 0:
                print(i, np.sum(rewards))   
            rew,obs = self.get_reward_obs(state)
            self.expand_graph(state)
            self.update_node_attributes(state,obs)
            self.evaluate_V()
            print(i,state, "%d %.3f" %(self.L.nodes[state]['n'],self.L.nodes[state]['p']))
            state = self.get_next_state(state)
            
            self.update_environment_graph(state,rew,obs)

            states += [state]
            rewards += [rew]
            observations += [obs]
            
        return states,rewards,observations