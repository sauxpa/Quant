#!/usr/bin/env python
# coding: utf-8

from ito_diffusion import *
from ito_diffusion_1d import *
import numpy as np
from numpy import random as rd
import pandas as pd
import abc

# Diffusion sheaf
# This is an attempt to create a sheaf of diffusion paths. Loosely speaking, from a given realization of a diffusion, the aim is to create a sequence of paths that share the same statistical properties as the original path while being "continous deformations" of it. That can be achieved by using "similar" gaussian increments at a given time during the discretization scheme for each path.
 
# In practice, the original path is a realization of a diffusion $dX_t = b(t,X_t)dt + \sigma(t,X_t)dW_t$. Any path generated in the sheaf follows the same SDE $dX'_t = b(t,X'_t)dt + \sigma(t,X'_t)dW'_t$ driven by a white noise $dW'$ which is chosen to be a mixture of the original noise $dW$ and an idiosyncratic Gaussian perturbation $\epsilon$ : $dW'_t = \alpha dW_t + \epsilon$, where $\alpha$ is the mixing coefficient. To ensure $dX_t$ remains driven by Brownian increments, $dW'_t$ needs to be distributed as $N(0,dt)$, which yields the noise-mixing relation $Var(\epsilon)=(1-\alpha^2)dx$.

# Note that this is indeed equivalent to sampling diffusion paths driven by Brownian motions with correlation $\alpha$ with the original process.

class Ito_diffusion_sheaf(Ito_diffusion):
    """Generic class for a sheaf of Ito diffusions.
    """
    def __init__(self, 
                 x0: float=0.0, 
                 T: float=1.0, 
                 scheme_steps: int=100, 
                 n_paths: int=10, 
                 path_mixing: float=0.99,
                 barrier=None, 
                 barrier_condition=None
                ) -> None:
        Ito_diffusion.__init__(self, 
                               x0=x0, 
                               T=T, 
                               scheme_steps=scheme_steps,
                               barrier=barrier, 
                               barrier_condition=barrier_condition
                              )
        self._n_paths = n_paths
        self._path_mixing = np.float(path_mixing)
    
    @property
    def n_paths(self) -> int:
        return self._n_paths
    @n_paths.setter
    def n_paths(self, new_n_paths: int) -> int:
        self._n_paths = float(new_n_paths)
    
    @property
    def path_mixing(self) -> float:
        return self._path_mixing
    @path_mixing.setter
    def path_mixing(self, new_path_mixing: float) -> None:
        self._path_mixing = float(new_path_mixing)

    @property
    def path_noise_stddev(self) -> float:
        return np.sqrt(1-self.path_mixing**2)
        
    def simulate(self) -> pd.DataFrame:
        """Euler-Maruyama scheme
        """
        paths = dict()
        gaussian_inc = rd.randn(self.scheme_steps)
        for n in range(self.n_paths):
            last_step = self.x0
            x = [last_step]
            for i, t in enumerate(self.time_steps[1:]):
                z = rd.randn() * self.path_noise_stddev
                previous_step = last_step
                last_step += self.drift(t, last_step) * self.scheme_step                 + self.vol(t, last_step) * self.scheme_step_sqrt * ( gaussian_inc[i] + z )
                
                if self.barrier_condition == 'absorb'                and self.barrier != None                and self.barrier_crossed(previous_step, last_step, self.barrier):
                    last_step = self.barrier
                
                x.append(last_step)
            paths['path {}'.format(n)] = x
            
        df = pd.DataFrame(paths)
        df.index = self.time_steps
        return df

class BM_sheaf(Ito_diffusion_sheaf, BM):
    """Instantiate Ito_diffusion_sheaf to simulate a sheaf of drifted Brownian motions
    dX_t = drift*dt + vol*dW_t
    where drift and vol are real numbers
    """
    def __init__(self,
                 x0: float=0.0, 
                 T: float=1.0, 
                 scheme_steps: int=100, 
                 n_paths: int=10, 
                 path_mixing: float=0.99,
                 drift: float=0.0, 
                 vol: float=1.0, 
                 barrier=None, 
                 barrier_condition=None
                ) -> None:
        Ito_diffusion_sheaf.__init__(self, 
                                     x0=x0, 
                                     T=T, 
                                     scheme_steps=scheme_steps,
                                     n_paths=n_paths, 
                                     path_mixing=path_mixing,
                                     barrier=barrier, 
                                     barrier_condition=barrier_condition
                                    )
        BM.__init__(self, 
                    x0=x0, 
                    T=T, 
                    scheme_steps=scheme_steps, 
                    drift=drift, 
                    vol=vol,
                    barrier=barrier,
                    barrier_condition=barrier_condition
                   )

class GBM_sheaf(Ito_diffusion_sheaf, GBM):
    """Instantiate Ito_diffusion to simulate a sheaf of geometric Brownian motions
    dX_t = drift*X_t*dt + vol*X_t*dW_t
    where drift and vol are real numbers
    """
    def __init__(self, 
                 x0: float=0.0, 
                 T: float=1.0, 
                 scheme_steps: int=100, 
                 n_paths: int=10, 
                 path_mixing: float=0.99,
                 drift: float=0.0, 
                 vol: float=1.0, 
                 barrier=None, 
                 barrier_condition=None
                ):
        Ito_diffusion_sheaf.__init__(self, 
                                     x0=x0, 
                                     T=T,
                                     scheme_steps=scheme_steps,
                                     n_paths=n_paths, 
                                     path_mixing=path_mixing,
                                     barrier=barrier, 
                                     barrier_condition=barrier_condition
                                    )
        GBM.__init__(self, 
                     x0=x0, 
                     T=T, 
                     scheme_steps=scheme_steps, 
                     drift=drift, 
                     vol=vol,
                     barrier=barrier,
                     barrier_condition=barrier_condition
                    )

class Vasicek_sheaf(Ito_diffusion_sheaf, Vasicek):
    """Instantiate Ito_diffusion to simulate a sheaf of mean-reverting Vasicek diffusion
    dX_t = mean_reversion*(long_term-X_t)*dt + vol*dW_t
    where mean_reversion, long_term and vol are real numbers
    """
    def __init__(self,
                x0: float=0.0, 
                 T: float=1.0, 
                 scheme_steps: int=100, 
                 n_paths: int=10, 
                 path_mixing: float=0.99,
                 mean_reversion: float=1.0,
                 long_term: float=0.0,
                 vol: float=1.0,
                 barrier=None,
                 barrier_condition=None
                ) -> None:
        Ito_diffusion_sheaf.__init__(self, 
                                     x0, 
                                     T, 
                                     scheme_steps,
                                     n_paths=n_paths, 
                                     path_mixing=path_mixing,
                                     barrier=barrier, 
                                     barrier_condition=barrier_condition
                                    )
        Vasicek.__init__(self, 
                         x0=x0, 
                         T=T, 
                         scheme_steps=scheme_steps,
                         mean_reversion=mean_reversion, 
                         long_term=long_term, 
                         vol=vol,
                         barrier=barrier, 
                         barrier_condition=barrier_condition
                        )

class CIR_sheaf(Ito_diffusion_sheaf, CIR):
    """Instantiate Ito_diffusion to simulate a sheaf of mean-reverting CIR diffusion
    dX_t = mean_reversion*(long_term-X_t)*dt + vol*sqrt(X_t)*dW_t
    where mean_reversion, long_term and vol are real numbers
    """
    def __init__(self,
                 x0: float=0.0, 
                 T: float=1.0, 
                 scheme_steps: int=100, 
                 n_paths: int=10, 
                 path_mixing: float=0.99,
                 mean_reversion: float=1.0, 
                 long_term: float=0.0, 
                 vol: float=1.0,
                 barrier=None, 
                 barrier_condition=None
                ) -> None:
        Ito_diffusion_sheaf.__init__(self, 
                                     x0, 
                                     T, 
                                     scheme_steps,
                                     n_paths=n_paths,
                                     path_mixing=path_mixing,
                                     barrier=barrier,
                                     barrier_condition=barrier_condition
                                    )
        CIR.__init__(self, 
                     x0=x0, 
                     T=T, 
                     scheme_steps=scheme_steps,
                     mean_reversion=mean_reversion,
                     long_term=long_term,
                     vol=vol,
                     barrier=barrier,
                     barrier_condition=barrier_condition
                    )

class pseudo_GBM_sheaf(Ito_diffusion_sheaf, pseudo_GBM):
    """Instantiate Ito_diffusion to simulate a sheaf of
    dX_t = drift*dt + vol*X_t*dW_t
    where r and vol are real numbers
    """
    def __init__(self, 
                 x0: float=0.0, 
                 T: float=1.0, 
                 scheme_steps: int=100, 
                 n_paths: int=10,
                 path_mixing: float=0.99,
                 drift: float=0.0, 
                 vol: float=1.0, 
                 barrier=None,
                 barrier_condition=None
                ) -> None:
        Ito_diffusion_sheaf.__init__(self, 
                                     x0, 
                                     T, 
                                     scheme_steps,
                                     n_paths=n_paths, 
                                     path_mixing=path_mixing,
                                     barrier=barrier, 
                                     barrier_condition=barrier_condition
                                    )
        pseudo_GBM.__init__(self, 
                            x0=x0, 
                            T=T, 
                            scheme_steps=scheme_steps,
                            drift=drift, 
                            vol=vol,
                            barrier=barrier, 
                            barrier_condition=barrier_condition
                           )

class Pinned_diffusion_sheaf(Ito_diffusion_sheaf, Pinned_diffusion):
    """Generic class for a sheaf of pinned diffusions, i.e diffusions which are constrained to arrive
    at a given point at the terminal date.
    """
    def __init__(self, 
                 x0: float=0.0, 
                 T: float=1.0, 
                 scheme_steps: int=100, 
                 n_paths: int=10, 
                 path_mixing: float=0.99,
                 pin: float=0.0,
                 vol: float=1.0,
                ) -> None:
        Ito_diffusion_sheaf.__init__(self, 
                                     x0, 
                                     T, 
                                     scheme_steps,
                                     n_paths=n_paths, 
                                     path_mixing=path_mixing
                                    )
        Pinned_diffusion.__init__(self, 
                                  x0=x0, 
                                  T=T,
                                  scheme_steps=scheme_steps,
                                  vol=vol, 
                                  pin=pin
                                 )

class Alpha_pinned_BM_sheaf(Pinned_diffusion_sheaf, Alpha_pinned_BM):
    """Instantiate Pinned_diffusion_sheaf to simulate a sheaf of alpha-pinned Brownian motion
    dX_t = alpha*(y-X_t)/(T-t)*dt + vol*dW_t
    where alpha, y (pin) and vol are real numbers
    """
    def __init__(self, 
                 x0: float=0.0, 
                 T: float=1.0, 
                 scheme_steps: int=100, 
                 n_paths: int=10, 
                 path_mixing: float=0.99,
                 alpha: float=1.0,
                 vol: float=1.0,
                 pin: float=0.0,
                ) -> None:
        Pinned_diffusion_sheaf.__init__(self, 
                                        x0=x0, 
                                        T=T, 
                                        scheme_steps=scheme_steps, 
                                        pin=pin,
                                        n_paths=n_paths,
                                        path_mixing=path_mixing
                                       )
        Alpha_pinned_BM.__init__(self, 
                                 x0=x0,
                                 T=T,
                                 scheme_steps=scheme_steps,
                                 alpha=alpha, 
                                 vol=vol,
                                 pin=pin
                                )

class F_pinned_BM_sheaf(Pinned_diffusion_sheaf, F_pinned_BM):
    """Instantiate Pinned_diffusion_sheaf to simulate a sheaf of F-pinned Brownian motions
    dX_t = f(t)*(y-X_t)/(1-F(t))*dt + sqrt(f(t))*dW_t
    where y (pin) is a real number, f and F respectively the pdf and cdf
    of a probability distribution over [0,T]
    """
    def __init__(self, 
                 x0: float=0.0, 
                 T: float=1.0, 
                 scheme_steps: int=100, 
                 n_paths: int=10, 
                 path_mixing: float=0.99,
                 distr=None, 
                 pin: float=0.0,
                ) -> None:
        Pinned_diffusion_sheaf.__init__(self, 
                                        x0=x0, 
                                        T=T, 
                                        scheme_steps=scheme_steps, 
                                        pin=pin,
                                        n_paths=n_paths, 
                                        path_mixing=path_mixing
                                       )
        F_pinned_BM.__init__(self, 
                             x0=x0,
                             T=T,
                             scheme_steps=scheme_steps,
                             distr=distr,
                             pin=pin
                            )