#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import abc


# ## Generic Ito diffusion
# 𝑑𝑋𝑡=𝑏(𝑡,𝑋𝑡)𝑑𝑡+𝜎(𝑡,𝑋𝑡)𝑑𝑊𝑡

# In[2]:


class Ito_diffusion(abc.ABC):
    """Generic class for Ito diffusion
    dX_t = b(t,X_t)dt + sigma(t,X_t)*dW_t
    with a potential boundary condition at barrier.
    Typical example : barrier=0, barrier_condition='absorb'
    (only this one is supported for now)
    """
    def __init__(self, x0=0, T=1, scheme_steps=100,                 barrier=None, barrier_condition=None):
        self._x0 = x0
        self._T = T
        self._scheme_steps = scheme_steps
        self._barrier = barrier
        self._barrier_condition = barrier_condition
        
    @property
    def x0(self):
        return self._x0
    @x0.setter
    def x0(self, new_x0):
        self._x0 = new_x0
    
    @property
    def T(self):
        return self._T
    @T.setter
    def T(self, new_T):
        self._T = new_T
    
    @property
    def scheme_steps(self):
        return self._scheme_steps
    @scheme_steps.setter
    def scheme_steps(self, new_scheme_steps):
        self.scheme_steps = new_scheme_steps
    
    @property
    def barrier(self):
        return self._barrier
    @barrier.setter
    def barrier(self, new_barrier):
        self._barrier = barrier
    
    @property
    def barrier_condition(self):
        if self._barrier_condition not in [ None, 'absorb']:
            raise NameError("Unsupported barrier condition : {}"                            .format(self._barrier_condition))
        else:
            return self._barrier_condition
    @barrier_condition.setter
    def barrier_condition(self, new_barrier_condition):
        self._barrier_condition = barrier_condition
        
    @property
    def scheme_step(self):
        return self.T/self.scheme_steps
    
    @property
    def scheme_step_sqrt(self):
        return np.sqrt(self.scheme_step)
   
    @property
    def time_steps(self):
        return [ step*self.scheme_step for step in range(self.scheme_steps+1) ]
    
    def barrier_crossed(self, x, y, barrier):
        """barrier is crossed if x and y are on each side of the barrier
        """
        return (x<=barrier and y>=barrier) or (x>=barrier and y<=barrier)
    
    @abc.abstractmethod
    def drift(self, t, x):
        pass
    
    @abc.abstractmethod
    def vol(self, t, x):
        pass
    
    @abc.abstractmethod
    def simulate(self):
        pass

