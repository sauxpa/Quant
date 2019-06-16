#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import random as rd
from scipy import integrate
import pandas as pd
import abc
from functools import lru_cache


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


# ## 1d diffusions

# In[3]:


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


# In[4]:


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


# In[5]:


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


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


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


# In[10]:


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


# In[11]:


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


# In[12]:


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


# ## Multidimensional diffusions

# In[13]:


class Ito_diffusion_multi_d(Ito_diffusion):
    """ Generic class for multidimensional Ito diffusion
    x0, drift and vol can be supplied as list/np.array...
    they will be casted to np.array
    x0 : initial vector, the dimension d of which is used to infer the dimension of the diffusion
    keys: optional, list of string of size d to name each of the dimension
    n_factors : number of factors i.e of Brownian motion driving the diffusion
    The covariance function has to return a matrix of dimension d*n_factors
    Potential boundary condition at barrier=(x1,...,xd).
    Example syntax : barrier=(0, None) means the boundary condition is on the first
    coordinate only, at 0.
    """
    def __init__(self, x0=np.zeros(1), T=1, scheme_steps=100, n_factors=1, keys=None,                 barrier=np.full(1, None), barrier_condition=np.full(1, None)):
        x0 = np.array(x0)
        super().__init__(x0=x0, T=T, scheme_steps=scheme_steps,                        barrier=barrier, barrier_condition=barrier_condition)
        if not keys:
            keys = ['dim {}'.format(i) for i in range(self.d)]
        self._keys = keys
        self._n_factors = n_factors
        
    @property
    def d(self):
        return len(self.x0)
    
    def simulate(self):
        """Euler-Maruyama scheme
        """
        last_step = self.x0
        x = [last_step]
        for t in self.time_steps[1:]:
            # z drawn for a N(0_d,1_d)
            previous_step = last_step
            z = rd.randn(self._n_factors)
            inc = self.drift(t, last_step) * self.scheme_step             + np.dot(self.vol(t, last_step), self.scheme_step_sqrt * z)
            last_step = last_step + inc
            
            if self.barrier_condition == 'absorb':
                for i, coord in enumerate(last_step):
                    if self.barrier[i] != None                    and self.barrier_crossed(previous_step[i], coord, self.barrier[i]):
                        last_step[i] = self.barrier[i]
                
            x.append(last_step)
        
        x = np.array(x)
        df_dict = dict()
        for i, key in enumerate(self._keys):
            df_dict[key] = x[:,i]
        df = pd.DataFrame(df_dict)
        df.index = self.time_steps
        return df


# In[14]:


class BM_multi_d(Ito_diffusion_multi_d):
    """Instantiate Ito_diffusion to simulate a drifted Brownian motion
    dX_t = drift*dt + vol*dW_t
    where drift and vol are real vector and matrix respectively
    """
    def __init__(self, x0=np.zeros(1), T=1, scheme_steps=100, drift=np.zeros(1), vol=np.eye(1),                 keys=None, barrier=np.full(1, None), barrier_condition=np.full(1, None)):
        self._drift_vector = np.array(drift)
        self._vol_matrix = np.array(vol) # vol is actually a covariance matrix here
        n_factors = self._vol_matrix.shape[1]
        super().__init__(x0=x0, T=T, scheme_steps=scheme_steps, keys=keys, n_factors=n_factors,                         barrier=barrier, barrier_condition=barrier_condition)
    
    @property
    def drift_vector(self):
        return self._drift_vector
    @drift_vector.setter
    def drift_vector(self, new_drift):
        self._drift_vector = np.array(new_drift)
    
    @property
    def vol_matrix(self):
        return self._vol_matrix
    @vol_matrix.setter
    def vol_matrix(self, new_vol):
        self._vol_matrix = np.array(new_vol)
        
    def drift(self, t, x):
        return self.drift_vector
    
    def vol(self, t, x):
        return self.vol_matrix


# In[15]:


class GBM_multi_d(Ito_diffusion_multi_d):
    """Instantiate Ito_diffusion to simulate a geometric Brownian motion
    dX_t = drift*X_t*dt + vol*X_t*dW_t
    where drift and vol are real vector and matrix respectively
    """
    def __init__(self, x0=np.ones(1), T=1, scheme_steps=100, drift=np.zeros(1), vol=np.eye(1),                 keys=None, barrier=np.full(1, None), barrier_condition=np.full(1, None)):
        self._drift_vector = np.array(drift)
        self._vol_matrix = np.array(vol) # vol is actually a covariance matrix here
        n_factors = self._vol_matrix.shape[1]
        super().__init__(x0=x0, T=T, scheme_steps=scheme_steps, n_factors=n_factors, keys=keys,                         barrier=barrier, barrier_condition=barrier_condition)
    
    @property
    def drift_vector(self):
        return self._drift_vector
    @drift_vector.setter
    def drift_vector(self, new_drift):
        self._drift_vector = np.array(new_drift)
    
    @property
    def vol_matrix(self):
        return self._vol_matrix
    @vol_matrix.setter
    def vol_matrix(self, new_vol):
        self._vol_matrix = np.array(new_vol)
        
    def drift(self, t, x):
        return np.multiply(x, self.drift_vector)
    
    def vol(self, t, x):
        return np.multiply(x,self.vol_matrix.T).T


# In[16]:


class SABR(Ito_diffusion_multi_d):
    """Instantiate Ito_diffusion to simulate a SABR stochastic vol model
    dX_t = s_t*X_t^beta*dW_t
    ds_t = vov*s_t*dB_t
    d<W,B>_t = rho*dt
    where beta, vov, rho are real numbers
    """
    def __init__(self, x0=[1,1], T=1, scheme_steps=100, keys=None,                 beta=1, vov=1, rho=0,                 barrier=np.full(1, None), barrier_condition=np.full(1, None)): 
        self._beta = np.float(beta)
        self._vov = np.float(vov)
        self._rho = np.float(rho)
        n_factors = 2
        super().__init__(x0=x0, T=T, scheme_steps=scheme_steps, n_factors=n_factors, keys=keys,                        barrier=barrier, barrier_condition=barrier_condition)
    
    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self, new_beta):
        self._beta = float(new_beta)
    
    @property
    def rho(self):
        return self._rho
    @rho.setter
    def rho(self, new_rho):
        self._rho = new_rho
     
    @property
    def vov(self):
        return self._vov
    @vov.setter
    def vov(self, new_vov):
        self._vov = new_vov
    
    @property
    def rho_dual(self):
        return np.sqrt(1-self.rho**2)
        
    def drift(self, t, x):
        return np.zeros_like(x)
    
    def vol(self, t, x):
        """Project dB onto dW and an orhtogonal white noise dZ
        dB_t = rho*dW_t + sqrt(1-rho^2)*dZ_t
        """
        return np.array([[x[1]*(x[0])**self.beta, 0],                         [self.vov*x[1]*self.rho, self.vov*x[1]*self.rho_dual]])


# In[17]:


class SABR_tanh(Ito_diffusion_multi_d):
    """Instantiate Ito_diffusion to simulate a modified SABR with tanh local vol model
    dX_t = s_t*C(X_t)*dW_t
    ds_t = vov*s_t*dB_t
    d<W,B>_t = rho*dt
    C(x) = tanh((x+shift)/l)
    where shift, l, vov, rho are real numbers
    """
    def __init__(self, x0=[1,1], T=1, scheme_steps=100, keys=None,                 shift=0, l=1, vov=1, rho=0,                 barrier=np.full(1, None), barrier_condition=np.full(1, None)): 
        self._shift = np.float(shift)
        self._l = np.float(l)
        self._vov = np.float(vov)
        self._rho = np.float(rho)
        n_factors = 2
        super().__init__(x0=x0, T=T, scheme_steps=scheme_steps, n_factors=n_factors, keys=keys,                        barrier=barrier, barrier_condition=barrier_condition)
        
    @property
    def shift(self):
        return self._shift
    @shift.setter
    def shift(self, new_shift):
        self._shift = float(new_shift)
    
    @property
    def l(self):
        return self._l
    @l.setter
    def l(self, new_l):
        self._l = float(new_l)
        
    @property
    def rho(self):
        return self._rho
    @rho.setter
    def rho(self, new_rho):
        self._rho = new_rho
     
    @property
    def vov(self):
        return self._vov
    @vov.setter
    def vov(self, new_vov):
        self._vov = new_vov
    
    @property
    def rho_dual(self):
        return np.sqrt(1-self.rho**2)
        
    def drift(self, t, x):
        return np.zeros_like(x)
    
    def vol(self, t, x):
        """Project dB onto dW and an orhtogonal white noise dZ
        dB_t = rho*dW_t + sqrt(1-rho^2)*dZ_t
        """
        return np.array([[x[1]*np.tanh((x[0]+self.shift)/self.l), 0],                         [self.vov*x[1]*self.rho, self.vov*x[1]*self.rho_dual]])


# ## Diffusion sheaf
# This is an attempt to create a sheaf of diffusion paths. Loosely speaking, from a given realization of a diffusion, the aim is to create a sequence of paths that share the same statistical properties as the original path while being "continous deformations" of it. That can be achieved by using "similar" gaussian increments at a given time during the discretization scheme for each path.
# 
# In practice, the original path is a realization of a diffusion $dX_t = b(t,X_t)dt + \sigma(t,X_t)dW_t$. Any path generated in the sheaf follows the same SDE $dX'_t = b(t,X'_t)dt + \sigma(t,X'_t)dW'_t$ driven by a white noise $dW'$ which is chosen to be a mixture of the original noise $dW$ and an idiosyncratic Gaussian perturbation $\epsilon$ : $dW'_t = \alpha dW_t + \epsilon$, where $\alpha$ is the mixing coefficient. To ensure $dX_t$ remains driven by Brownian increments, $dW'_t$ needs to be distributed as $N(0,dt)$, which yields the noise-mixing relation $Var(\epsilon)=(1-\alpha^2)dx$.
# 
# Note that this is indeed equivalent to sampling diffusion paths driven by Brownian motions with correlation $\alpha$ with the original process.

# In[18]:


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


# In[19]:


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


# In[20]:


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


# In[21]:


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


# In[22]:


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


# In[23]:


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


# In[24]:


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


# In[25]:


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


# In[26]:


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


# ## Multifractal diffusions

# In[27]:


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
        self._x0 = new_x0
        
    @property
    def T(self):
        return self._T
    @T.setter
    def T(self, new_T):
        self._T = new_T
        
    @property
    def scheme_step(self):
        return self._scheme_step
    @scheme_step.setter
    def scheme_step(self, new_scheme_step):
        self.scheme_step = new_scheme_step
    
    @property
    def intermittency(self):
        return self._intermittency
    @intermittency.setter
    def intermittency(self, new_intermittency):
        self._intermittency = new_intermittency
    
    @property
    def integral_scale(self):
        return self._integral_scale
    @integral_scale.setter
    def integral_scale(self, new_integral_scale):
        self._integral_scale = new_integral_scale
    
    @property
    def l(self):
        return self._l
    @l.setter
    def l(self, new_l):
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


# In[ ]:




