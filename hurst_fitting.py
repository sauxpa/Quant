#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from statsmodels.graphics.tsaplots import acf
from scipy.optimize import minimize_scalar


# In[2]:


class Hurst_fitter():
    """Estimate Hurst parameter of a time series by fitting the autocorrelogram
    associated with a fractional Gaussian noise to the empirical autocorrelogram.
    """
    def __init__(self, n_lags=10, df=None):
        self._n_lags = n_lags
        self._df = df
        
    @property
    def n_lags(self):
        return self._n_lags
    @n_lags.setter
    def n_lags(self, new_n_lags):
        self._n_lags = new_n_lags
    
    @property
    def df(self):
        return self._df
    @df.setter
    def df(self, new_df):
        self._df = new_df
        
    @property
    def lag_range(self):
        return range(self.n_lags+1)
    
    @property
    def df_inc(self):
        return (self.df-self.df.shift(1)).iloc[1:]

    def autocorr_frac_noise(self, H, lag):
        return 0.5*(np.abs(lag+1)**(2*H)+np.abs(lag-1)**(2*H)-2*np.abs(lag)**(2*H))
    
    def autocorr_frac_noise_range(self, H):
        return [self.autocorr_frac_noise(H,x) for x in self.lag_range]
    
    def objective_func(self, H):
        ys_fit = self.autocorr_frac_noise_range(H)
        ys = acf(self.df_inc, nlags=self.n_lags)
        return np.linalg.norm(ys-ys_fit)
    
    def fit(self):
        fit_info = minimize_scalar(self.objective_func, method='brent')        
        if fit_info.success:
            return fit_info.x
        else:
            raise Exception('Failed calibration...')

