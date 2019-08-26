#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root


# In[2]:


ONE_BP = 1e-4
ONE_PCT = 1e-2


# In[5]:


class Fitter():
    """Wrapper for a model and a dataframe of market quoted volatilies.
    Calibrates the parameters of the model to fit the vol surface.
    Names of the parameters are inferred from init_params_map
    """
    def __init__(self, model=None, f=None, T_expiry=None,                 init_params_map=dict(), df=None, tenor='1Y'):
        self._model = model
        self._f = f
        self._T_expiry = T_expiry
        self._init_params_map = init_params_map
        self._df = df
        self._tenor = tenor
        self._sol = None

        self._model.f = f
        self._model.T_expiry = T_expiry
    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, new_model):
        self._model = new_model

    @property
    def f(self):
        return self._f
    @f.setter
    def f(self, new_f):
        self._f = new_f
        self.model.f = new_f
    
    @property
    def T_expiry(self):
        return self._T_expiry
    @T_expiry.setter
    def T_expiry(self, new_T_expiry):
        self._T_expiry = new_T_expiry
        self.model.T_expiry = new_T_expiry
        
    @property
    def init_params_map(self):
        return self._init_params_map
    @init_params_map.setter
    def init_params_map(self, new_init_params_map):
        self._init_params_map = new_init_params_map
    
    @property
    def df(self):
        return self._df
    @df.setter
    def df(self, new_df):
        self._df = new_df
        
    @property
    def tenor(self):
        return self._tenor
    @tenor.setter
    def tenor(self, new_tenor):
        self._tenor = new_tenor
    
    @property
    def strike_map(self):
        ATM = self.f
        return dict(zip(self.df[self.tenor].index,                        [self.f + K*ONE_BP if K != 'ATM'                         else self.f                         for K in self.df[self.tenor].index]))
    
    @property
    def market_smile(self):
        return dict(zip(self.strike_map.values(),                        list(map(lambda x: x * ONE_BP, self.df[self.tenor].values))))
    
    @property
    def sol(self):
        return self._sol
    
    def calibrate_model_from_params_map(self, params_map):
        for param_name, param_value in params_map.items():
            setattr(self.model, param_name, param_value)
    
    def calibrate_model(self):
        self.calibrate_model_from_params_map(self.sol)
        return self.model
        
    def constraints_func(self, params):
        params_map = dict(zip(self.init_params_map.keys(), params))
        self.calibrate_model_from_params_map(params_map)
        return self.model.smile_func
        
    def all_constraints_func(self, params):
        constr = []
        smile_func = self.constraints_func(params)
        for strike, implied_vol in self.market_smile.items():
            constr.append(smile_func(strike)-implied_vol)
        return( constr )
    
    def fit(self, verbose=False):
        sol = root(self.all_constraints_func,                   list(self.init_params_map.values()), method = 'lm')
        if verbose:
            print('Success!') if sol.success else print('Failure!')
            print(sol.message)
        self._sol = dict(zip(self.init_params_map.keys(), sol.x))


# In[ ]:




