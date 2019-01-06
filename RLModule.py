
# coding: utf-8

# In[1]:


import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
import operator
import abc


# # Generic class for agent

# In[2]:


class Agent():
    """Agent:
    Object with a state, the history of past state, and total accumulated reward
    """
    def __init__(self, state=None):
        self._state = state
        self._state_history = [state] if state != None else []
        self._total_reward = 0
        
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, new_state):
        self._state = new_state
        self._state_history.append(new_state)
    
    def reset(self, state=None):
        self._state = state
        self.resetHistory()
        
    def resetHistory(self):
        self._state_history = [];
    
    def summary(self):
        print('Current state : {}'.format(self._state))
        print('State history : {}'.format(self._state_history))
        print('Total reward  : {}'.format(self._total_reward))
        
    def move(self, next_state, reward):
        self._state = next_state
        self._state_history.append(next_state)
        self._total_reward += reward


# # Generic class for environment

# An environment has : 
# 1. a list of state 
# 2. a action function that maps each state to the set of possible next states
# 3. a reward function that maps each (action,state) to a reward
# 4. a terminal state
# 
# Those are defined here as abstract method to be specified in each environment

# In[3]:


class Environment(abc.ABC):
    """Environment:
    Contains the state space, the (state,action)->new_state mapping, (state,action,new_state)->reward mapping
    One of the states is the terminal state}
    """
    def __init__(self):
        self._state_space = []
    
    @abc.abstractmethod
    def actionStateMap(self, state):
        pass
    
    @abc.abstractmethod
    def getReward(self, state=None, action=None, next_state=None):
        pass

    @abc.abstractmethod
    def isTerminal(self, state=None):
        pass
    
    def getNexStateFromAction(self, state=None, action=None):
        next_state = self.actionStateMap(state).get(action)
        reward = self.getReward(state, action, next_state)
        return next_state, reward
    
    def summary(self):
        print('Terminal state: {}'.format(self._terminal_state))
        for state in self._state_space:
            print('{}'.format(state))
            actions = self.actionStateMap(state)
            for action, next_state in actions.items():
                print('\t{} --> {}'.format(action, next_state))


# # Generic gridworld

# Gridworld is a generic environment where the state space is chessboard-like

# In[4]:


class Gridworld(Environment):
    """Chessboard-like environment
    """
    def __init__(self, n_h, n_v, terminal_state=None):
        super().__init__()
        self._n_h = n_h # number of horizontal cells
        self._n_v = n_v # number of vertical cells
        self._terminal_state = terminal_state
        self._directions = {
            'right': [1, 0],
            'left': [-1, 0],
            'up': [0, 1],
            'down': [0, -1],
        }
        # states are tuples (i,j) representing cells of the grid
        self._state_space = [(i,j) for i in range(n_h) for j in range(n_v)]
            
    def resetAgentState(self, Agent, state=None):
        # random reset, except if a state is specified
        if state != None:
            Agent.reset(state=state)
        else:
            i = rd.choice(range(len(self._state_space)))
            Agent.reset(state=self._state_space[i])
    
    def isTerminal(self, state=None):
        return state == self._terminal_state
        
    def isLeftWall(self, state):
        return state[0] <= 0
    
    def isRightWall(self, state):
        return state[0] >= self._n_h-1
    
    def isDownWall(self, state):
        return state[1] <= 0

    def isUpWall(self, state):
        return state[1] >= self._n_v-1
    
    def printPath(self, state_history, reward_history):
        grid = np.full((self._n_h, self._n_v), np.nan)
        for state, reward in zip(list(state_history),list(reward_history)):
            grid[state] = reward
                                 
        np.set_printoptions(nanstr='.')
        # need to flip the h-axis since grid is a matrix
        print(np.flip(grid.T,0))
        print('Total reward : {}'.format(sum(reward_history)))


# In BasicGridworld, you can access all available right/left/up/down squares, with a reward of -1 for each step

# In[5]:


class BasicGridworld(Gridworld):
    """Simple grid, only up/down/right/left actions
    """
    def __init__(self, n_h, n_v, terminal_state=None):
        super().__init__(n_h, n_v, terminal_state)
        
    def actionStateMap(self, state):
        directions = self._directions.copy()
         
        if self.isLeftWall(state):
            del directions['left']
        
        if self.isRightWall(state):
            del directions['right']
        
        if self.isUpWall(state):
            del directions['up']
        
        if self.isDownWall(state):
            del directions['down']
        
        # lambda to wrap the transition from state using up/down/left/right
        move_lambda = lambda direction: tuple(np.array(state)+np.array(direction))
        
        # returns list of possible actions, list of possible next_states
        return dict(zip(list(directions.keys()), list(map(move_lambda, list(directions.values())))))

    def getReward(self, state=None, action=None, next_state=None):
        return -1


# In WindyGridworld, there's an upward wind blowing through the middle of the grid

# In[6]:


class WindyGridworld(Gridworld):
    """Simple grid, the vertical median line has a wind effect that adds an extra up:
    """
    def __init__(self, n_h, n_v, terminal_state=None):
        super().__init__(n_h, n_v, terminal_state)
        
    def actionStateMap(self, state):
        directions = self._directions.copy()
        
        if self.isLeftWall(state):
            del directions['left']
        
        if self.isRightWall(state):
            del directions['right']
        
        if self.isUpWall(state):
            del directions['up']
        
        if self.isDownWall(state):
            del directions['down']
        
        # if state is on the vertical median line of the grid, each move is affected by an extra up (if possible)
        if state[0] == int(self._n_h/2):
            for key, direction in directions.items():
                potential_next_state = tuple(np.array(state)+np.array(direction))
                if not self.isUpWall(potential_next_state):
                    directions[key] = tuple(np.array(direction)+np.array([0, 1])) 
            
        # lambda to wrap the transition from state using up/down/left/right
        move_lambda = lambda direction: tuple(np.array(state)+np.array(direction))
        
        # returns list of possible actions, list of possible next_states
        return dict(zip(list(directions.keys()), list(map(move_lambda, list(directions.values())))))
        
    def getReward(self, state=None, action=None, next_state=None):
        return -1


# in CliffGridWorld, there is a cliff at the bottom of the grid : fall and you start over with a massive negative reward!

# In[7]:


class CliffGridworld(Gridworld):
    """Simple grid, the bottom row of is a cliff that penalizes the agent if it falls over it
    """
    def __init__(self, n_h, n_v, startover_state=None, terminal_state=None):
        super().__init__(n_h, n_v, terminal_state)
        # initial_state is where the agent is brought back if it falls in the cliff
        self._startover_state = startover_state
        
    def printCliff(self):
        grid = np.full((self._n_h, self._n_v), '.')
        for i in range(1,self._n_h-1):
            grid[(i, 0)] = '*'
        # need to flip the h-axis since grid is a matrix
        print(np.flip(grid.T,0))
        
    # overload summary to add a print of the cliff
    def summary(self):
        self.printCliff()
        super().summary()
        
    # if right above the cliff and the action is 'down' or if at the bottom-left corner and the action is 'right'
    def fallsInHole(self, state, action):
        return ((state[1] == 1 and state[0] not in [0, self._n_h-1]) and action == 'down') or (state == (0, 0) and action == 'right')
            
    def actionStateMap(self, state):
        directions = self._directions.copy()
        
        if self.isLeftWall(state):
            del directions['left']
        
        if self.isRightWall(state):
            del directions['right']
        
        if self.isUpWall(state):
            del directions['up']
        
        if self.isDownWall(state):
            del directions['down']
           
        for key, direction in directions.items():
            if self.fallsInHole(state, key):
                directions[key] = tuple(np.array(self._startover_state)-np.array(state))
                
        # lambda to wrap the transition from state using up/down/left/right
        move_lambda = lambda direction: tuple(np.array(state)+np.array(direction))
        
        # returns list of possible actions, list of possible next_states
        return dict(zip(list(directions.keys()), list(map(move_lambda, list(directions.values())))))
        
    def getReward(self, state=None, action=None, next_state=None):
        if self.fallsInHole(state, action):
            return -100
        else:
            return -1


# # Policy

# In[8]:


def greedyPickAction(state, Q):
    """Greedy policy:
    for a given state, returns the action that maximizes the current estimate of the Q function
    """
    return max(Q[state].items(), key=operator.itemgetter(1))[0]


# In[9]:


class Policy():
    """Policy:
    Contains a state->action mapping
    """
    def __init__(self, Environment):
        self._Environment = Environment
    
    @abc.abstractmethod
    def pickAction(self, state=None, context={}):
        pass


# In[10]:


class EpsilonGreedyPolicy(Policy):
    """Epsilon Greedy policy:
    for a given state, returns the action that maximizes the current estimate of the Q function
    with probability 1-epsilion, otherwise randomly draw an action
    """
    def __init__(self, Environment, epsilon):
        super().__init__(Environment)
        self._epsilon = epsilon
        
    def pickAction(self, state=None, context={}):
        if rd.uniform() <= 1-self._epsilon: 
            return greedyPickAction(state, context.get('Q'))
        else:
            return rd.choice(list(self._Environment.actionStateMap(state).keys()))


# In[11]:


class RandomPolicy(EpsilonGreedyPolicy):
    """Random policy:
    for a given state, returns an action drawn uniformly at random
    """
    def __init__(self, Environment):
        super().__init__(Environment, 1)


# In[12]:


class GreedyPolicy(EpsilonGreedyPolicy):
    """Greedy policy:
    for a given state, returns the action that maximizes the current estimate of the Q function
    """
    def __init__(self, Environment):
        super().__init__(Environment, 0)


# # Learning solver

# In[13]:


class ActionValueRL(abc.ABC):
    """ActionValueRL:
        Performs policy optimization using action-value learning
    """
    def __init__(self, Agent=None, Environment=None, params={}):
        self._Agent = Agent
        self._Environment = Environment
        self._state_space = Environment._state_space
        
        # Q is a dict with keys {state} and values dict {action: value}
        # To call Q(s,a): self._Q[state][action] 
        self._Q = dict(zip(self._Environment._state_space, [{},]*len(self._Environment._state_space)))
        self._max_count = int(1e5) # max number of steps per epoch
        self._iter_per_epoch = []
        
        self._policy_params = params.get('policy_params')
        self._agent_params = params.get('agent_params')

        # set Policy
        self._policy_name = ''
        self._Policy = None
        self.reset(self._policy_params, self._agent_params)
        
    # initialize to Q(s,a)=0
    def reset(self, policy_params={}, agent_params={}):
        for state in self._state_space:
            actions = self._Environment.actionStateMap(state)
            q_state = {}
            for action in actions:
                q_state[action] = 0
            self._Q[state] = q_state
                
        # if policy_params is not empty, reset policy as well
        if policy_params:
            self.setPolicy(policy_params)
        
        # if agent_params is not empty, reset agent as well
        if agent_params:
            self._Agent.reset(agent_params.get('state'))
    
    def getQ(self, state, action):
        return self._Q[state][action]
    
    def setQ(self, state, action, new_value):
        self._Q[state][action] = new_value
        
    def setPolicy(self, policy_params={}):
        self._policy_name = policy_params.get('policy_name')
        if self._policy_name == 'Random':
            self._Policy = RandomPolicy(self._Environment)
        elif self._policy_name == 'Greedy':
            self._Policy = GreedyPolicy(self._Environment)
        elif self._policy_name == 'EpsilonGreedy':
            self._Policy = EpsilonGreedyPolicy(self._Environment, policy_params.get('epsilon'))
        else:
            raise NameError( 'Policy name not found : {}'.format(self._policy_name) )
    
    # optimal path is the one where each action is selected using the greedy policy
    def optimalPath(self, initial_state):
        count, total_reward = 0, 0
        state = initial_state
        state_history = [initial_state]
        reward_history = [0]
        Greedy = GreedyPolicy(self._Environment)
        while not self._Environment.isTerminal(state) and count < self._max_count:
            #action = max(self._Q[state].items(), key=operator.itemgetter(1))[0]
            action = greedyPickAction(state, self._Q)
            state, reward = self._Environment.getNexStateFromAction(state, action)
            state_history.append(state)
            count += 1
            reward_history.append(reward)
            
        return state_history, reward_history
        
    @abc.abstractmethod
    def learn(self, n_epochs):
        pass    


# In[14]:


class Sarsa(ActionValueRL):
    """SARSA:
        SARSA implementation of on-policy TD
    """
    def __init__(self, Agent=None, Environment=None, params={}):
        super().__init__(Agent=Agent, Environment=Environment, params=params)
        self._learning_rate = params.get('learning_rate')
        self._discount_factor = params.get('discount_factor')
        
    def learn(self, n_epochs, init_state):
        for count_epoch in range(n_epochs):
            # initialize state
            self._Environment.resetAgentState(self._Agent, state=init_state)
            state = self._Agent.state
            
            # choose action from state using policy derived from Q
            action = self._Policy.pickAction(state, {'Q': self._Q})
            count = 0
            
            while not self._Environment.isTerminal(state) and count < self._max_count:
                # take action, observe next_state and reward
                next_state, reward = self._Environment.getNexStateFromAction(state, action)
                
                # choose next_action from next_state using policy derived from Q
                next_action = self._Policy.pickAction(next_state, {'Q': self._Q})
                                
                # compute TD target and update Q
                TD_target = reward+self._discount_factor*self.getQ(next_state, next_action)-self.getQ(state, action)
                new_value = self.getQ(state, action)+self._learning_rate*TD_target
                self.setQ(state, action, new_value)
                
                # move to next state, update Agent internal state
                state = next_state
                action = next_action
                self._Agent.move(next_state, reward)
                count += 1
            self._iter_per_epoch.append(count)


# In[15]:


class QLearning(ActionValueRL):
    """Q-learning:
        Q-learning implementation of off-policy TD -- learn the optimal policy directly while following a different exploration policy
    """
    def __init__(self, Agent=None, Environment=None, params={}):
        super().__init__(Agent=Agent, Environment=Environment, params=params)
        self._learning_rate = params.get('learning_rate')
        self._discount_factor = params.get('discount_factor')
        
    def learn(self, n_epochs, init_state):
        for count_epoch in range(n_epochs):
            # initialize state
            self._Environment.resetAgentState(self._Agent, state=init_state)
            state = self._Agent.state
            
            count = 0
            
            while not self._Environment.isTerminal(state) and count < self._max_count:
                 # choose action from state using policy derived from Q
                action = self._Policy.pickAction(state, {'Q': self._Q})
            
                # take action, observe next_state and reward
                next_state, reward = self._Environment.getNexStateFromAction(state, action)
                                                
                # off-policy learning : take the max of the Q function at the current state, regardless of the action followed by the agent
                Q_max = max(self._Q[next_state].items(), key=operator.itemgetter(1))[1]
                
                # compute TD target and update Q
                TD_target = reward+self._discount_factor*Q_max-self.getQ(state, action)
                new_value = self.getQ(state, action)+self._learning_rate*TD_target
                self.setQ(state, action, new_value)
                
                # move to next state, update Agent internal state
                state = next_state
                self._Agent.move(next_state, reward)
                count += 1
            self._iter_per_epoch.append(count)

