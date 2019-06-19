#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ito_diffusion import *
import numpy as np
from numpy import random as rd
import pandas as pd
import abc
from functools import lru_cache


# ## 1d diffusions

# In[2]:


class Ito_diffusion_1d(Ito_diffusion):
    def __init__(self, x0=0, T=1, scheme_steps=100, barrier=None, barrier_condition=None):
        super().__init__(x0=x0, T=T, scheme_steps=scheme_steps,                         barrier=barrier, barrier_condition=barrier_condition)
    
    def simulate(self):
        """Euler-Maruyama scheme
        """
        last_step = self.x0
        x = [last_step]
        for t in self.time_steps[1:]:
            # z drawn for a N(0,1)
            z = rd.randn()
            previous_step = last_step
            last_step += self.drift(t, last_step) * self.scheme_step             + self.vol(t, last_step) * self.scheme_step_sqrt * z
            
            if self.barrier_condition == 'absorb'            and self.barrier != None            and self.barrier_crossed(previous_step, last_step, self.barrier):
                last_step = self.barrier
        
            x.append(last_step)
            
        df = pd.DataFrame({'spot': x})
        df.index = self.time_steps
        return df


# In[3]:


class BM(Ito_diffusion_1d):
    """Instantiate Ito_diffusion to simulate a drifted Brownian motion
    dX_t = drift*dt + vol*dW_t
    where drift and vol are real numbers
    """
    def __init__(self, x0=0, T=1, scheme_steps=100, drift=0, vol=1,                 barrier=None, barrier_condition=None):
        super().__init__(x0=x0, T=T, scheme_steps=scheme_steps,                        barrier=barrier, barrier_condition=barrier_condition)
        self._drift_double = np.float(drift)
        self._vol_double = np.float(vol)
    
    @property
    def drift_double(self):
        return self._drift_double
    @drift_double.setter
    def drift_double(self, new_drift):
        self._drift_double = new_drift
    
    @property
    def vol_double(self):
        return self._vol_double
    @vol_double.setter
    def vol_double(self, new_vol):
        self._vol_double = new_vol
        
    def drift(self, t, x):
        return self.drift_double
    
    def vol(self, t, x):
        return self.vol_double


# In[4]:


class GBM(Ito_diffusion_1d):
    """Instantiate Ito_diffusion to simulate a geometric Brownian motion
    dX_t = drift*X_t*dt + vol*X_t*dW_t
    where drift and vol are real numbers
    """
    def __init__(self, x0=1, T=1, scheme_steps=100, drift=0, vol=1,                 barrier=None, barrier_condition=None):
        super().__init__(x0, T, scheme_steps,                         barrier=barrier, barrier_condition=barrier_condition)
        self._drift_double = np.float(drift)
        self._vol_double = np.float(vol)
    
    @property
    def drift_double(self):
        return self._drift_double
    @drift_double.setter
    def drift_double(self, new_drift):
        self._drift_double = new_drift
    
    @property
    def vol_double(self):
        return self._vol_double
    @vol_double.setter
    def vol_double(self, new_vol):
        self._vol_double = new_vol
        
    def drift(self, t, x):
        return self.drift_double * x
    
    def vol(self, t, x):
        return self.vol_double * x


# In[5]:


class SLN(Ito_diffusion_1d):
    """Instantiate Ito_diffusion to simulate a shifted lognormal diffusion
    dX_t = drift*X_t*dt + sigma*(shift*X_t+(mixing-shift)*X_0+(1-m)*L)*dW_t
    where drift and vol are real numbers
    """
    def __init__(self, x0=1, T=1, scheme_steps=100,                 drift=0, sigma=0, shift=0, mixing=0, L=0,                 barrier=None, barrier_condition=None):
        super().__init__(x0, T, scheme_steps,                         barrier=barrier, barrier_condition=barrier_condition)
        self._drift_double = drift
        self._sigma = sigma
        self._shift = shift
        self._mixing = mixing
        self._L = L    
        
    @property
    def drift_double(self):
        return self._drift_double
    @drift_double.setter
    def drift_double(self, new_drift):
        self._drift_double = new_drift
    
    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self, new_sigma):
        self._sigma = float(new_sigma)
    
    @property
    def shift(self):
        return self._shift
    @shift.setter
    def shift(self, new_shift):
        self._shift = new_shift
     
    @property
    def mixing(self):
        return self._mixing
    @mixing.setter
    def mixing(self, new_mixing):
        self._mixing = new_mixing
        
    @property
    def L(self):
        return self._L
    @L.setter
    def L(self, new_L):
        self._L = new_L
        
    def drift(self, t, x):
        return self.drift_double * x
    
    def vol(self, t, x):
        return self.sigma*(self.shift*x+(self.mixing-self.shift)*self.x0+(1-self.mixing)*self.L)


# In[6]:


class Vasicek(Ito_diffusion_1d):
    """Instantiate Ito_diffusion to simulate a mean-reverting Vasicek diffusion
    dX_t = mean_reversion*(long_term-X_t)*dt + vol*dW_t
    where mean_reversion, long_term and vol are real numbers
    """
    def __init__(self, x0=1, T=1, scheme_steps=100, mean_reversion=1, long_term=0, vol=1,                barrier=None, barrier_condition=None):
        super().__init__(x0, T, scheme_steps,                         barrier=barrier, barrier_condition=barrier_condition)
        self._mean_reversion = np.float(mean_reversion)
        self._long_term = np.float(long_term)
        self._vol_double = np.float(vol)
    
    @property
    def mean_reversion(self):
        return self._mean_reversion
    @mean_reversion.setter
    def mean_reversion(self, new_mean_reversion):
        self._mean_reversion = new_mean_reversion
    
    @property
    def long_term(self):
        return self._long_term
    @long_term.setter
    def long_term(self, new_long_term):
        self._long_term = new_long_term

    @property
    def vol_double(self):
        return self._vol_double
    @vol_double.setter
    def vol_double(self, new_vol):
        self._vol_double = new_vol
        
    def drift(self, t, x):
        return self.mean_reversion * (self.long_term-x)
    
    def vol(self, t, x):
        return self.vol_double


# In[7]:


class CIR(Ito_diffusion_1d):
    """Instantiate Ito_diffusion to simulate a mean-reverting CIR diffusion
    dX_t = mean_reversion*(long_term-X_t)*dt + vol*sqrt(X_t)*dW_t
    where mean_reversion, long_term and vol are real numbers
    """
    def __init__(self, x0=1, T=1, scheme_steps=100, mean_reversion=1, long_term=0, vol=1,                barrier=None, barrier_condition=None):
        super().__init__(x0, T, scheme_steps,                         barrier=barrier, barrier_condition=barrier_condition)
        self._mean_reversion = np.float(mean_reversion)
        self._long_term = np.float(long_term)
        self._vol_double = np.float(vol)
    
    @property
    def mean_reversion(self):
        return self._mean_reversion
    @mean_reversion.setter
    def mean_reversion(self, new_mean_reversion):
        self._mean_reversion = new_mean_reversion
    
    @property
    def long_term(self):
        return self._long_term
    @long_term.setter
    def long_term(self, new_long_term):
        self._long_term = new_long_term

    @property
    def vol_double(self):
        return self._vol_double
    @vol_double.setter
    def vol_double(self, new_vol):
        self._vol_double = new_vol

    @property
    def feller_condition(self):
        """Returns whether mean_reversion * long_term > 0.5*vol^2
        """
        return self.mean_reversion * self.long_term > 0.5*self.vol_double**2

    def drift(self, t, x):
        return self.mean_reversion * (self.long_term-x)
    
    def vol(self, t, x):
        return self.vol_double * np.sqrt(x)


# In[8]:


class pseudo_GBM(Ito_diffusion_1d):
    """Instantiate Ito_diffusion to simulate
    dX_t = drift*dt + vol*X_t*dW_t
    where r and vol are real numbers
    """
    def __init__(self, x0=1, T=1, scheme_steps=100, drift=0, vol=1,                barrier=None, barrier_condition=None):
        super().__init__(x0, T, scheme_steps,                         barrier=barrier, barrier_condition=barrier_condition)
        self._drift_double = np.float(drift)
        self._vol_double = np.float(vol)
    
    @property
    def drift_double(self):
        return self._drift_double
    @drift_double.setter
    def drift_double(self, new_drift):
        self._drift_double = new_drift

    @property
    def vol_double(self):
        return self._vol_double
    @vol_double.setter
    def vol_double(self, new_vol):
        self._vol_double = new_vol
        
    def drift(self, t, x):
        return self.drift_double
    
    def vol(self, t, x):
        return self.vol_double*x


# In[9]:


class Pinned_diffusion(Ito_diffusion_1d):
    """Generic class for pinned diffusions, i.e diffusions which are constrained to arrive
    at a given point at the terminal date.
    """
    def __init__(self, x0=0, T=1, scheme_steps=100, alpha=1, vol=1, pin=0):
        super().__init__(x0=x0, T=T, scheme_steps=scheme_steps)
        self._pin = np.float(pin)
    
    @property
    def pin(self):
        return self._pin
    @pin.setter
    def pin(self, new_pin):
        self._pin = new_pin
    
    @abc.abstractmethod
    def _h(self, t):
        pass
    
    def drift(self, t, x):
        """In a pinned diffusion, drift is written as
        h(t)*(y-x)
        """
        if( t == self.T ):
            return 0
        else:
            return self._h(t) * (self.pin - x)
        
    def simulate(self, show_backbone=True):
        """Euler-Maruyama scheme
        """
        df = super().simulate()
        df.loc[self.T].spot = self.pin
        
        if show_backbone:
            # discretization scheme to estimate the backbone
            last_step = self.x0
            backbone = [last_step]
            for t in self.time_steps[1:]:
                last_step += self._h(t) * (self.pin-last_step) * self.scheme_step
                backbone.append(last_step)
            df['backbone'] = backbone
        
        return df


# In[10]:


class Alpha_pinned_BM(Pinned_diffusion):
    """Instantiate Ito_diffusion to simulate an alpha-pinned Brownian motion
    dX_t = alpha*(y-X_t)/(T-t)*dt + vol*dW_t
    where alpha, y (pin) and vol are real numbers
    """
    def __init__(self, x0=0, T=1, scheme_steps=100, alpha=1, vol=1, pin=0):
        super().__init__(x0=x0, T=T, scheme_steps=scheme_steps, pin=pin)
        self._alpha = np.float(alpha)
        self._vol_double = np.float(vol)
    
    @property
    def alpha(self):
        return self._alpha
    @alpha.setter
    def alpha(self, new_alpha):
        self._alpha = new_alpha
        
    @property
    def vol_double(self):
        return self._vol_double
    @vol_double.setter
    def vol_double(self, new_vol):
        self._vol_double = new_vol
        
    def _h(self, t):
        if( t == self.T ):
            return 0
        else:
            return self.alpha / (self.T-t)
    
    def vol(self, t, x):
       return self.vol_double


# In[11]:


class F_pinned_BM(Pinned_diffusion):
    """Instantiate Ito_diffusion to simulate an F-pinned Brownian motion
    dX_t = f(t)*(y-X_t)/(1-F(t))*dt + sqrt(f(t))*dW_t
    where y (pin) is a real number, f and F respectively the pdf and cdf
    of a probability distribution over [0,T]
    """
    def __init__(self, x0=0, T=1, scheme_steps=100, distr=None, pin=0):
        super().__init__(x0=x0, T=T, scheme_steps=scheme_steps, pin=pin)
        if not distr:
            raise NameError( "Must specify a probability distribution" )
        else:
            self._distr = distr
            self._f = distr.pdf
            self._F = distr.cdf
        
    def _h(self, t):
        if( t == self._T ):
            return 0
        else:
            return self._f(t) / (1-self._F(t))
        
    def vol(self, t, x):
        return np.sqrt(self._f(t))


# ## Multifractal diffusions

# In[12]:


class Lognormal_multifractal():
    """Simulate a lognormal multifractal cascade process
    X_t = lim_{l->0+} int_0^t exp(w_l,T(u))dW_u
    where w_l,T(u) is a gaussian process with known mean and covariance functions.
    This is approximated by discretization of dX_t = exp(w_l,T(t))dW_t for a small
    value of l.
    """
    def __init__(self, x0=0, T=1, scheme_step=0.01,                 intermittency=0.01, integral_scale=1, l=None):
        self._x0 = np.float(x0)
        self._T = np.float(T)
        self._scheme_step = np.float(scheme_step)
        self._intermittency = np.float(intermittency)
        self._integral_scale = np.float(integral_scale)
        
        if not l:
            l = self._scheme_step / 128
        self._l = l
        
    @property
    def x0(self):
        return self._x0
    @x0.setter
    def x0(self, new_x0):
        self.omega_simulate.cache_clear()
        self._x0 = new_x0
        
    @property
    def T(self):
        return self._T
    @T.setter
    def T(self, new_T):
        self.omega_simulate.cache_clear()
        self._T = new_T
        
    @property
    def scheme_step(self):
        return self._scheme_step
    @scheme_step.setter
    def scheme_step(self, new_scheme_step):
        self.omega_simulate.cache_clear()
        self.scheme_step = new_scheme_step
    
    @property
    def intermittency(self):
        return self._intermittency
    @intermittency.setter
    def intermittency(self, new_intermittency):
        self.omega_simulate.cache_clear()
        self._intermittency = new_intermittency
    
    @property
    def integral_scale(self):
        return self._integral_scale
    @integral_scale.setter
    def integral_scale(self, new_integral_scale):
        self.omega_simulate.cache_clear()
        self._integral_scale = new_integral_scale
    
    @property
    def l(self):
        return self._l
    @l.setter
    def l(self, new_l):
        self.omega_simulate.cache_clear()
        self._l = new_l
        
    @property
    def l_sqrt(self):
        return np.sqrt(self.l)
    
    @property
    def time_steps(self):
        return [ step*self.l for step in range(self.scheme_steps+1) ] 
    
    @property
    def scheme_steps(self):
        return np.floor(self.T / self.l).astype('int')
    
    def expectation(self):
        return -self.intermittency**2 * (np.log(self.integral_scale / self.l) - 1)

    def covariance(self, tau):
        if self.integral_scale <= tau:
            return 0
        elif tau < self.integral_scale and tau >= self.l:
            return self.intermittency ** 2 * np.log(self.integral_scale / tau)
        else:
            return self.intermittency ** 2 * (np.log(self.integral_scale / self.l)                                               + 1 - tau / self.l)

    def covariance_matrix(self):
        cov = np.zeros((self.scheme_steps, self.scheme_steps))
        for i in range(self.scheme_steps):
            cov[i][i] = self.covariance(0)
            for j in range(i):
                cov[i][j] = self.covariance((i - j) * self._scheme_step)
                cov[j][i] = cov[i][j]
        return cov
    
    @lru_cache(maxsize=None)
    def omega_simulate(self):
        return rd.multivariate_normal(self.expectation() * np.ones((self.scheme_steps,)),                                      self.covariance_matrix())
        
    def MRM(self):
        """Multifractal Random Measure
        """
        last_step = 0
        x = [last_step]
        omega = self.omega_simulate()
        for n in range(1, self.scheme_steps):
            last_step += self.l * np.exp(2 * omega[n - 1])
            x.append(last_step)
            
    def simulate(self):
        """Multifractal Random Walk
        """
        last_step = self.x0 
        x = [last_step]
        last_brownian = self.x0
        BM = [last_brownian]
        omega = self.omega_simulate()
        for om in omega:
            noise = rd.randn() * self.l_sqrt
            last_step += np.exp(om) * noise
            x.append(last_step)
            last_brownian += noise
            BM.append(last_brownian) 
        
        df = pd.DataFrame({'MRW': x, 'BM': BM})
        df.index = self.time_steps
        return df

