#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ito_diffusion import *
import numpy as np
from numpy import random as rd
import pandas as pd
import abc


# ## Diffusion sheaf
# This is an attempt to create a sheaf of diffusion paths. Loosely speaking, from a given realization of a diffusion, the aim is to create a sequence of paths that share the same statistical properties as the original path while being "continous deformations" of it. That can be achieved by using "similar" gaussian increments at a given time during the discretization scheme for each path.
# 
# In practice, the original path is a realization of a diffusion $dX_t = b(t,X_t)dt + \sigma(t,X_t)dW_t$. Any path generated in the sheaf follows the same SDE $dX'_t = b(t,X'_t)dt + \sigma(t,X'_t)dW'_t$ driven by a white noise $dW'$ which is chosen to be a mixture of the original noise $dW$ and an idiosyncratic Gaussian perturbation $\epsilon$ : $dW'_t = \alpha dW_t + \epsilon$, where $\alpha$ is the mixing coefficient. To ensure $dX_t$ remains driven by Brownian increments, $dW'_t$ needs to be distributed as $N(0,dt)$, which yields the noise-mixing relation $Var(\epsilon)=(1-\alpha^2)dx$.
# 
# Note that this is indeed equivalent to sampling diffusion paths driven by Brownian motions with correlation $\alpha$ with the original process.

# In[2]:


class Ito_diffusion_sheaf(Ito_diffusion):
    """Generic class for a sheaf of Ito diffusions.
    """
    def __init__(self, x0=0, T=1, scheme_steps=100, n_paths=10, path_mixing=0.99,                 barrier=None, barrier_condition=None):
        super().__init__(x0=x0, T=T, scheme_steps=scheme_steps,                         barrier=barrier, barrier_condition=barrier_condition)
        self._n_paths = n_paths
        self._path_mixing = np.float(path_mixing)
    
    @property
    def n_paths(self):
        return self._n_paths
    @n_paths.setter
    def n_paths(self, new_n_paths):
        self._n_paths = float(new_n_paths)
    
    @property
    def path_mixing(self):
        return self._path_mixing
    @path_mixing.setter
    def path_mixing(self, new_path_mixing):
        self._path_mixing = float(new_path_mixing)
    
    @property
    def path_noise_stddev(self):
        return np.sqrt(1-self.path_mixing**2)
        
    def simulate(self):
        """Euler-Maruyama scheme
        """
        paths = dict()
        gaussian_inc = rd.randn(self.scheme_steps)
        for n in range(self.n_paths):
            last_step = self.x0
            x = [last_step]
            for i, t in enumerate(self.time_steps[1:]):
                z = rd.randn() * self.path_noise_stddev
                last_step += self.drift(t, last_step) * self.scheme_step                 + self.vol(t, last_step) * self.scheme_step_sqrt * ( gaussian_inc[i] + z )
                x.append(last_step)
            paths['path {}'.format(n)] = x
            
        df = pd.DataFrame(paths)
        df.index = self.time_steps
        return df


# In[3]:


class BM_sheaf(Ito_diffusion_sheaf):
    """Instantiate Ito_diffusion_sheaf to simulate a sheaf of drifted Brownian motions
    dX_t = drift*dt + vol*dW_t
    where drift and vol are real numbers
    """
    def __init__(self, x0=0, T=1, scheme_steps=100, n_paths=10, path_mixing=0.99,                 drift=0, vol=1, barrier=None, barrier_condition=None):
        super().__init__(x0=x0, T=T, scheme_steps=scheme_steps,                         n_paths=n_paths, path_mixing=path_mixing,                         barrier=barrier, barrier_condition=barrier_condition)
        self._drift_double = np.float(drift)
        self._vol_double = np.float(vol)
        
    def drift(self, t, x):
        return self._drift_double
    
    def vol(self, t, x):
        return self._vol_double


# In[4]:


class GBM_sheaf(Ito_diffusion_sheaf):
    """Instantiate Ito_diffusion to simulate a sheaf of geometric Brownian motions
    dX_t = drift*X_t*dt + vol*X_t*dW_t
    where drift and vol are real numbers
    """
    def __init__(self, x0=0, T=1, scheme_steps=100, n_paths=10, path_mixing=0.99,                 drift=0, vol=1, barrier=None, barrier_condition=None):
        super().__init__(x0=x0, T=T, scheme_steps=scheme_steps,                         n_paths=n_paths, path_mixing=path_mixing,                         barrier=barrier, barrier_condition=barrier_condition)
        self._drift_double = np.float(drift)
        self._vol_double = np.float(vol)
        
    def drift(self, t, x):
        return self._drift_double * x
    
    def vol(self, t, x):
        return self._vol_double * x


# In[5]:


class Vasicek_sheaf(Ito_diffusion_sheaf):
    """Instantiate Ito_diffusion to simulate a sheaf of mean-reverting Vasicek diffusion
    dX_t = mean_reversion*(long_term-X_t)*dt + vol*dW_t
    where mean_reversion, long_term and vol are real numbers
    """
    def __init__(self, x0=1, T=1, scheme_steps=100, n_paths=10, path_mixing=0.99,                 mean_reversion=1, long_term=0, vol=1,                 barrier=None, barrier_condition=None):
        super().__init__(x0, T, scheme_steps,                         n_paths=n_paths, path_mixing=path_mixing,                         barrier=barrier, barrier_condition=barrier_condition)
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


# In[6]:


class CIR_sheaf(Ito_diffusion_sheaf):
    """Instantiate Ito_diffusion to simulate a sheaf of mean-reverting CIR diffusion
    dX_t = mean_reversion*(long_term-X_t)*dt + vol*sqrt(X_t)*dW_t
    where mean_reversion, long_term and vol are real numbers
    """
    def __init__(self, x0=1, T=1, scheme_steps=100, n_paths=10, path_mixing=0.99,                 mean_reversion=1, long_term=0, vol=1,                 barrier=None, barrier_condition=None):
        super().__init__(x0, T, scheme_steps, n_paths=n_paths, path_mixing=path_mixing,                         barrier=barrier, barrier_condition=barrier_condition)
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


# In[7]:


class pseudo_GBM_sheaf(Ito_diffusion_sheaf):
    """Instantiate Ito_diffusion to simulate a sheaf of
    dX_t = drift*dt + vol*X_t*dW_t
    where r and vol are real numbers
    """
    def __init__(self, x0=1, T=1, scheme_steps=100, drift=0, vol=1,                 n_paths=10, path_mixing=0.99,                 barrier=None, barrier_condition=None):
        super().__init__(x0, T, scheme_steps, n_paths=n_paths, path_mixing=path_mixing,                         barrier=barrier, barrier_condition=barrier_condition)
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


# In[8]:


class Pinned_diffusion_sheaf(Ito_diffusion_sheaf):
    """Generic class for a sheaf of pinned diffusions, i.e diffusions which are constrained to arrive
    at a given point at the terminal date.
    """
    def __init__(self, x0=0, T=1, scheme_steps=100, alpha=1, vol=1, pin=0,                n_paths=10, path_mixing=0.99):
        super().__init__(x0=x0, T=T, scheme_steps=scheme_steps,                         n_paths=n_paths, path_mixing=path_mixing)
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
        if( t == self._T ):
            return 0
        else:
            return self._h(t) * (self.pin - x)
        
    def simulate(self, show_backbone=True):
        """Euler-Maruyama scheme
        """
        df = super().simulate()
        for n in range(self._n_paths):
            df.loc[self._T]['path {}'.format(n)] = self.pin
        
        if show_backbone:
            # discretization scheme to estimate the backbone
            last_step = self.x0
            backbone = [last_step]
            for t in self.time_steps[1:]:
                last_step += self._h(t) * (self.pin-last_step) * self.scheme_step
                backbone.append(last_step)
            df['backbone'] = backbone

        return df


# In[9]:


class Alpha_pinned_BM_sheaf(Pinned_diffusion_sheaf):
    """Instantiate Pinned_diffusion_sheaf to simulate a sheaf of alpha-pinned Brownian motion
    dX_t = alpha*(y-X_t)/(T-t)*dt + vol*dW_t
    where alpha, y (pin) and vol are real numbers
    """
    def __init__(self, x0=0, T=1, scheme_steps=100, alpha=1, vol=1, pin=0,                n_paths=10, path_mixing=0.99):
        super().__init__(x0=x0, T=T, scheme_steps=scheme_steps, pin=pin,                        n_paths=n_paths, path_mixing=path_mixing)
        self._alpha = np.float(alpha)
        self._vol_double = np.float(vol)
        
    def _h(self, t):
        if( t == self._T ):
            return 0
        else:
            return self._alpha / (self._T-t)
    
    def vol(self, t, x):
        return self._vol_double


# In[10]:


class F_pinned_BM_sheaf(Pinned_diffusion_sheaf):
    """Instantiate Pinned_diffusion_sheaf to simulate a sheaf of F-pinned Brownian motions
    dX_t = f(t)*(y-X_t)/(1-F(t))*dt + sqrt(f(t))*dW_t
    where y (pin) is a real number, f and F respectively the pdf and cdf
    of a probability distribution over [0,T]
    """
    def __init__(self, x0=0, T=1, scheme_steps=100, distr=None, pin=0,                n_paths=10, path_mixing=0.99):
        super().__init__(x0=x0, T=T, scheme_steps=scheme_steps, pin=pin,                        n_paths=n_paths, path_mixing=path_mixing)
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

