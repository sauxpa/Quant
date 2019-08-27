#!/usr/bin/env python
# coding: utf-8

from ito_diffusion import *
import numpy as np
from numpy import random as rd
import pandas as pd
import abc

# Multidimensional diffusions

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
    def __init__(self, 
                 x0: np.ndarray=np.zeros(1), 
                 T: float=1.0, 
                 scheme_steps: int=100, 
                 n_factors: int=1, 
                 keys=None,
                 barrier: np.ndarray=np.full(1, None), 
                 barrier_condition: np.ndarray=np.full(1, None)
                ) -> None:
        
        x0 = np.array(x0)
        super().__init__(x0=x0, 
                         T=T, 
                         scheme_steps=scheme_steps,
                         barrier=barrier,
                         barrier_condition=barrier_condition
                        )
        
        if not keys:
            keys = ['dim {}'.format(i) for i in range(self.d)]
        self._keys = keys
        self._n_factors = n_factors
        
    @property
    def d(self) -> int:
        return len(self.x0)
    
    def simulate(self) -> pd.DataFrame:
        """Euler-Maruyama scheme
        """
        last_step = self.x0
        x = [last_step]
        for t in self.time_steps[1:]:
            # z drawn for a N(0_d,1_d)
            previous_step = last_step
            z = rd.randn(self._n_factors)
            inc = self.drift(t, last_step) * self.scheme_step + np.dot(self.vol(t, last_step), self.scheme_step_sqrt * z)
            last_step = last_step + inc
            
            if self.barrier_condition == 'absorb':
                for i, coord in enumerate(last_step):
                    if self.barrier[i] != None\
                    and self.barrier_crossed(previous_step[i], coord, self.barrier[i]):
                        last_step[i] = self.barrier[i]
                
            x.append(last_step)
        
        x = np.array(x)
        df_dict = dict()
        for i, key in enumerate(self._keys):
            df_dict[key] = x[:,i]
        df = pd.DataFrame(df_dict)
        df.index = self.time_steps
        return df

class BM_multi_d(Ito_diffusion_multi_d):
    """Instantiate Ito_diffusion to simulate a drifted Brownian motion
    dX_t = drift*dt + vol*dW_t
    where drift and vol are real vector and matrix respectively
    """
    def __init__(self, 
                 x0: np.ndarray=np.zeros(1), 
                 T: float=1.0, 
                 scheme_steps: int=100, 
                 drift: np.ndarray=np.zeros(1), 
                 vol: np.ndarray=np.eye(1),
                 keys=None, 
                 barrier: np.ndarray=np.full(1, None),
                 barrier_condition: np.ndarray=np.full(1, None)
                ) -> None:
        self._drift_vector = np.array(drift)
        self._vol_matrix = np.array(vol) # vol is actually a covariance matrix here
        n_factors = self._vol_matrix.shape[1]
        super().__init__(x0=x0,
                         T=T,
                         scheme_steps=scheme_steps,
                         keys=keys,
                         n_factors=n_factors,
                         barrier=barrier, 
                         barrier_condition=barrier_condition
                        )
    
    @property
    def drift_vector(self) -> np.ndarray:
        return self._drift_vector
    @drift_vector.setter
    def drift_vector(self, new_drift: np.ndarray) -> None:
        self._drift_vector = np.array(new_drift)
    
    @property
    def vol_matrix(self) -> np.ndarray:
        return self._vol_matrix
    @vol_matrix.setter
    def vol_matrix(self, new_vol: np.ndarray) -> None:
        self._vol_matrix = np.array(new_vol)
        
    def drift(self, t, x) -> np.ndarray:
        return self.drift_vector
    
    def vol(self, t, x) -> np.ndarray:
        return self.vol_matrix

class GBM_multi_d(Ito_diffusion_multi_d):
    """Instantiate Ito_diffusion to simulate a geometric Brownian motion
    dX_t = drift*X_t*dt + vol*X_t*dW_t
    where drift and vol are real vector and matrix respectively
    """
    def __init__(self, 
                 x0: np.ndarray=np.ones(1), 
                 T: float=1.0, 
                 scheme_steps: int=100, 
                 drift: np.ndarray=np.zeros(1), 
                 vol: np.ndarray=np.eye(1),
                 keys=None,
                 barrier: np.ndarray=np.full(1, None),
                 barrier_condition: np.ndarray=np.full(1, None)
                ) -> None:
        self._drift_vector = np.array(drift)
        self._vol_matrix = np.array(vol) # vol is actually a covariance matrix here
        n_factors = self._vol_matrix.shape[1]
        super().__init__(x0=x0,
                         T=T,
                         scheme_steps=scheme_steps,
                         n_factors=n_factors,
                         keys=keys,
                         barrier=barrier,
                         barrier_condition=barrier_condition
                        )    
    @property
    def drift_vector(self) -> np.ndarray:
        return self._drift_vector
    @drift_vector.setter
    def drift_vector(self, new_drift: np.ndarray) -> np.ndarray:
        self._drift_vector = np.array(new_drift)
    
    @property
    def vol_matrix(self) -> np.ndarray:
        return self._vol_matrix
    @vol_matrix.setter
    def vol_matrix(self, new_vol: np.ndarray) -> None:
        self._vol_matrix = np.array(new_vol)
        
    def drift(self, t, x: np.ndarray) -> np.ndarray:
        return np.multiply(x, self.drift_vector)
    
    def vol(self, t, x: np.ndarray) -> np.ndarray:
        return np.multiply(x,self.vol_matrix.T).T

class SABR(Ito_diffusion_multi_d):
    """Instantiate Ito_diffusion to simulate a SABR stochastic vol model
    dX_t = s_t*X_t^beta*dW_t
    ds_t = vov*s_t*dB_t
    d<W,B>_t = rho*dt
    where beta, vov, rho are real numbers
    """
    def __init__(self, 
                 x0: np.ndarray=np.array([1,1]),
                 T: float=1.0, 
                 scheme_steps: int=100,
                 keys=None,
                 beta: float=1.0, 
                 vov: float=1.0,
                 rho: float=0.0,
                 barrier: np.ndarray=np.full(1, None), 
                 barrier_condition: np.ndarray=np.full(1, None)
                ) -> None: 
        self._beta = np.float(beta)
        self._vov = np.float(vov)
        self._rho = np.float(rho)
        n_factors = 2
        super().__init__(x0=x0, 
                         T=T, 
                         scheme_steps=scheme_steps,
                         n_factors=n_factors,
                         keys=keys,
                         barrier=barrier,
                         barrier_condition=barrier_condition
                        )
    
    @property
    def beta(self) -> float:
        return self._beta
    @beta.setter
    def beta(self, new_beta: float) -> None:
        self._beta = float(new_beta)
    
    @property
    def rho(self) -> float:
        return self._rho
    @rho.setter
    def rho(self, new_rho: float) -> None:
        self._rho = new_rho
        
    @property
    def vov(self) -> float:
        return self._vov
    @vov.setter
    def vov(self, new_vov: float) -> None:
        self._vov = new_vov
    
    @property
    def rho_dual(self) -> float:
        return np.sqrt(1-self.rho**2)
        
    def drift(self, t, x) -> np.ndarray:
        return np.zeros_like(x)
    
    def vol(self, t, x: np.ndarray) -> np.ndarray:
        """Project dB onto dW and an orhtogonal white noise dZ
        dB_t = rho*dW_t + sqrt(1-rho^2)*dZ_t
        """
        return np.array([[x[1]*(x[0])**self.beta, 0],                         [self.vov*x[1]*self.rho, self.vov*x[1]*self.rho_dual]])

class SABR_AS_lognorm(Ito_diffusion_multi_d):
    """Instantiate Ito_diffusion to simulate a modified SABR with local spiky vol
    dX_t = s_t*C(X_t)*dW_t
    ds_t = vov*s_t*dB_t
    d<W,B>_t = rho*dt
    C(x) = exp(-c*log((y+shift)/K_max)^2)
    where shift, K_max, c, vov, rho are real numbers
    """
    def __init__(self, 
                 x0: np.ndarray=np.array([1,1]), 
                 T: float=1.0, 
                 scheme_steps: int=100,
                 keys=None,
                 shift: float=0.0,
                 K_max: float=1.0,
                 c: float=1.0,
                 vov: float=1.0, 
                 rho: float=0.0,
                 barrier: np.ndarray=np.full(1, None),
                 barrier_condition: np.ndarray=np.full(1, None)
                ) -> None: 
        self._shift = np.float(shift)
        self._K_max = np.float(K_max)
        self._c = np.float(c)
        self._vov = np.float(vov)
        self._rho = np.float(rho)
        n_factors = 2
        super().__init__(x0=x0, 
                         T=T, 
                         scheme_steps=scheme_steps, 
                         n_factors=n_factors,
                         keys=keys, 
                         barrier=barrier, 
                         barrier_condition=barrier_condition
                        )
        
    @property
    def shift(self) -> float:
        return self._shift
    @shift.setter
    def shift(self, new_shift: float) -> None:
        self._shift = float(new_shift)
    
    @property
    def K_max(self) -> float:
        return self._K_max
    @K_max.setter
    def K_max(self, new_K_max: float) -> None:
        self._K_max = float(new_K_max)
    
    @property
    def c(self) -> float:
        return self._c
    @c.setter
    def c(self, new_c: float) -> None:
        self._c = float(new_c)
        
    @property
    def rho(self) -> float:
        return self._rho
    @rho.setter
    def rho(self, new_rho: float) -> None:
        self._rho = new_rho
        
    @property
    def vov(self) -> float:
        return self._vov
    @vov.setter
    def vov(self, new_vov: float) -> None:
        self._vov = new_vov
    
    @property
    def rho_dual(self) -> float:
        return np.sqrt(1-self.rho**2)
        
    def drift(self, t, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)
    
    def vol(self, t, x: np.ndarray) -> np.ndarray:
        """Project dB onto dW and an orhtogonal white noise dZ
        dB_t = rho*dW_t + sqrt(1-rho^2)*dZ_t
        """
        return np.array([[x[1]*np.exp(-self.c*np.log((x[0]+self.shift)/self.K_max)**2), 0],                         [self.vov*x[1]*self.rho, self.vov*x[1]*self.rho_dual]])

class SABR_AS_loglogistic(Ito_diffusion_multi_d):
    """Instantiate Ito_diffusion to simulate a modified SABR with local spiky vol
    dX_t = s_t*C(X_t)*dW_t
    ds_t = vov*s_t*dB_t
    d<W,B>_t = rho*dt
    C(x) = beta/alpha*(xs/alpha)^(beta-1)/(1+(*xs/alpha)^beta)^2
    xs = x + shift
    mode = alpha*((beta-1)/(beta+1))^(1-beta)
    where shift, mode, beta, vov, rho are real numbers
    """
    def __init__(self, 
                 x0: np.ndarray=np.array([1,1]), 
                 T: float=1.0, 
                 scheme_steps: int=100,
                 keys=None,
                 shift: float=0.0,
                 mode: float=1.0,
                 beta: float=1.0,
                 vov: float=1.0, 
                 rho: float=0.0,
                 barrier: np.ndarray=np.full(1, None),
                 barrier_condition: np.ndarray=np.full(1, None)
                ) -> None: 
        self._shift = np.float(shift)
        self._mode = np.float(mode)
        self._beta = np.float(beta)
        self._vov = np.float(vov)
        self._rho = np.float(rho)
        n_factors = 2
        super().__init__(x0=x0, 
                         T=T, 
                         scheme_steps=scheme_steps, 
                         n_factors=n_factors,
                         keys=keys, 
                         barrier=barrier, 
                         barrier_condition=barrier_condition
                        )
        
    @property
    def shift(self) -> float:
        return self._shift
    @shift.setter
    def shift(self, new_shift: float) -> None:
        self._shift = float(new_shift)
    
    @property
    def mode(self) -> float:
        return self._mode
    @mode.setter
    def mode(self, new_mode: float) -> None:
        self._mode = float(new_mode)
    
    @property
    def beta(self) -> float:
        return self._beta
    @beta.setter
    def beta(self, new_beta: float) -> None:
        self._beta = float(new_beta)
        
    @property
    def rho(self) -> float:
        return self._rho
    @rho.setter
    def rho(self, new_rho: float) -> None:
        self._rho = new_rho
        
    @property
    def vov(self) -> float:
        return self._vov
    @vov.setter
    def vov(self, new_vov: float) -> None:
        self._vov = new_vov
    
    @property
    def rho_dual(self) -> float:
        return np.sqrt(1-self.rho**2)
        
    def drift(self, t, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)
    
    def vol(self, t, x: np.ndarray) -> np.ndarray:
        """Project dB onto dW and an orhtogonal white noise dZ
        dB_t = rho*dW_t + sqrt(1-rho^2)*dZ_t
        """
        return np.array([[x[1]*np.exp(-self.c*np.log((x[0]+self.shift)/self.K_max)**2), 0],                         [self.vov*x[1]*self.rho, self.vov*x[1]*self.rho_dual]])
    
class SABR_tanh(Ito_diffusion_multi_d):
    """Instantiate Ito_diffusion to simulate a modified SABR with tanh local vol model
    dX_t = s_t*C(X_t)*dW_t
    ds_t = vov*s_t*dB_t
    d<W,B>_t = rho*dt
    C(x) = tanh((x+shift)/l)
    where shift, l, vov, rho are real numbers
    """
    def __init__(self, 
                 x0: np.ndarray=np.array([1,1]), 
                 T: float=1.0,
                 scheme_steps: int=100, 
                 keys=None,
                 shift: float=0.0,
                 l: float=1.0,
                 vov: float=1.0,
                 rho: float=0.0,
                 barrier: np.ndarray=np.full(1, None),
                 barrier_condition: np.ndarray=np.full(1, None)
                ) -> None: 
        self._shift = np.float(shift)
        self._l = np.float(l)
        self._vov = np.float(vov)
        self._rho = np.float(rho)
        n_factors = 2
        super().__init__(x0=x0, 
                         T=T, 
                         scheme_steps=scheme_steps, 
                         n_factors=n_factors, 
                         keys=keys,
                         barrier=barrier,
                         barrier_condition=barrier_condition
                        )
        
    @property
    def shift(self) -> float:
        return self._shift
    @shift.setter
    def shift(self, new_shift: np.ndarray) -> None:
        self._shift = float(new_shift)
    
    @property
    def l(self) -> float:
        return self._l
    @l.setter
    def l(self, new_l: float) -> None:
        self._l = float(new_l)
        
    @property
    def rho(self) -> float:
        return self._rho
    @rho.setter
    def rho(self, new_rho: float) -> None:
        self._rho = new_rho
        
    @property
    def vov(self) -> float:
        return self._vov
    @vov.setter
    def vov(self, new_vov: float) -> None:
        self._vov = new_vov
    
    @property
    def rho_dual(self) -> float:
        return np.sqrt(1-self.rho**2)
        
    def drift(self, t, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)
    
    def vol(self, t, x: np.ndarray) -> np.ndarray:
        """Project dB onto dW and an orhtogonal white noise dZ
        dB_t = rho*dW_t + sqrt(1-rho^2)*dZ_t
        """
        return np.array([[x[1]*np.tanh((x[0]+self.shift)/self.l), 0],                         [self.vov*x[1]*self.rho, self.vov*x[1]*self.rho_dual]])