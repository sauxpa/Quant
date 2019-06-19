#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import random as rd


# ## Fractional Gaussian noise
# 
# Covariance function of fractional Brownian motion :
# 
# E[B_H(t) B_H (s)]=1/2*(|t|^{2H}+|s|^{2H}-|t-s|^{2H})
# 
# Very slow implementation where the full trajectory of the fBM is generated as a correlated gaussian draw (requires manipulation of scheme_steps*scheme_steps matrix). 
# That'll do for now

# In[2]:


class Fractional_Gaussian_Noise():
    def __init__(self, H=0.5, T=1, scheme_steps=100):
        self._T = T
        self._scheme_steps = scheme_steps
        self._H = H
        
    @property
    def scheme_steps(self):
        return self._scheme_steps
    @scheme_steps.setter
    def scheme_steps(self, new_scheme_steps):
        self.scheme_steps = new_scheme_steps
    
    @property
    def T(self):
        return self._T    
    @T.setter
    def T(self, new_T):
        self.T = new_T
        
    @property
    def H(self):
        return self._H    
    @H.setter
    def H(self, new_H):
        self.H = new_H
    
    @property
    def scheme_step(self):
        return self.T/self.scheme_steps
    
    def covariance(self, s, t) :
        return 0.5*(t**(2*self.H)+s**(2*self.H)-np.abs(t-s)**(2*self.H))
        
    def covariance_matrix(self):
        cov = np.zeros((self.scheme_steps+1, self.scheme_steps+1))
        for i in range(self.scheme_steps+1):
            cov[i][i] = (i*self.scheme_step)**(2*self.H)
            for j in range(i):
                cov[i][j] = self.covariance(i*self.scheme_step, j*self.scheme_step)
                cov[j][i] = cov[i][j]
        return cov
    
    def simulate(self):
        """Returns increments of the fBM
        """
        cum_noise = rd.multivariate_normal(np.zeros((self.scheme_steps+1,)), self.covariance_matrix())
        return cum_noise[1:]-cum_noise[:-1]

