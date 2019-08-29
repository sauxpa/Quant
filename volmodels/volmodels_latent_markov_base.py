#!/usr/bin/env python
# coding: utf-8

import numpy as np
from volmodels import Vol_model
from scipy.stats import norm
from scipy.integrate import quad
from functools import partial
import abc

class Latent_Markov_Vol_simple(Vol_model):
    """Latent Markov volatility model.
    The process starts with volatility sigma_0 and randomly 
    jumps to sigma_1 with intensity lambda. The call pricing formula
    can be obtained by averaging standard pricing formulae over the
    exponential distribution of transition time from sigma_0 to sigma_1.
    Two marking modes are available:
    1) Intensity: directly mark the transition intensity
    2) Vov: this transition intensity can be interpreted as a vov by analogy with classical stochastic volatility models. Typically vov^2 * T = var(log(vol_t)), which gives the identification vov^2 = intensity*log(sigma_1/sigma_0)^2.
    """
    def __init__(self,
                 sigma_0: float=1.0, 
                 sigma_1: float=1.0, 
                 intensity=None, 
                 vov=None,
                 marking_mode='intensity',                 
                 f=None,
                 T_expiry: float=1.0,
                 vol_type=None,
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                ) -> None:
        
        super().__init__(f=f, 
                         T_expiry=T_expiry,
                         vol_type=vol_type,
                         logmoneyness_lo=logmoneyness_lo,
                         logmoneyness_hi=logmoneyness_hi,
                         K_lo=K_lo,
                         K_hi=K_hi,
                         strike_type=strike_type,
                         n_strikes=n_strikes
                        )
        self._sigma_0 = sigma_0
        self._sigma_1 = sigma_1
        self._marking_mode = marking_mode
        
        if marking_mode == 'intensity':
            self._intensity = intensity
            self._vov = np.sqrt(intensity) * np.abs(np.log(sigma_1/sigma_0))
        elif marking_mode == 'vov':
            self._vov = vov
            self._intensity = (self.vov/np.log(self.sigma_1/self.sigma_0))**2
        else:
            raise NameError("Unsupported marking mode : {}".format(self._marking_mode))
            
    @property
    def sigma_0(self) -> float:
        return self._sigma_0
    @sigma_0.setter
    def sigma_0(self, new_sigma_0: float) -> None:
        self._sigma_0 = new_sigma_0
    
    @property
    def sigma_1(self) -> float:
        return self._sigma_1
    @sigma_1.setter
    def sigma_1(self, new_sigma_1: float) -> None:
        self._sigma_1 = new_sigma_1
        
    @property
    def intensity(self) -> float:
        if self.marking_mode == 'intensity':
            return self._intensity
        else:
            return (self.vov/np.log(self.sigma_1/self.sigma_0))**2
    @intensity.setter
    def intensity(self, new_intensity: float) -> None:
        if self.marking_mode == 'intensity':
            self._intensity = new_intensity
        else:
            raise NameError("Cannot mark intensity in {} marking mode".format(self._marking_mode))
    
    @property
    def vov(self) -> float:
        if self.marking_mode == 'vov':
            return self._vov
        else:
            return np.sqrt(self.intensity) * np.abs(np.log(self.sigma_1/self.sigma_0))
    @vov.setter
    def vov(self, new_vov: float) -> None:
        if self.marking_mode == 'vov':
            self._vov = new_vov
        else:
            raise NameError("Cannot mark vov in {} marking mode".format(self._marking_mode))
            
    @property
    def marking_mode(self) -> str:
        if self._marking_mode not in ['intensity', 'vov']:
            raise NameError("Unsupported marking mode : {}".format(self._marking_mode))
        return self._marking_mode
    @marking_mode.setter
    def marking_mode(self, new_marking_mode: str) -> None:
        self._marking_mode = new_marking_mode
        
    @abc.abstractmethod
    def smile_func(self, K: float) -> float:
        pass
    
    def total_std(self, t: float) -> float:
        """Total standard deviation between 0 and T_expiry with a vol switch at t
        """
        return np.sqrt(self.sigma_1**2*self.T_expiry + (self.sigma_0**2-self.sigma_1**2)*t)
    
    @property
    def total_std_0(self) -> float:
        """Total standard deviation between 0 and T_expiry assuming no vol switch
        """
        return self.sigma_0*np.sqrt(self.T_expiry)
    
    @abc.abstractmethod
    def integrand(self, K: float, t: float) -> float:
        pass
    
    @abc.abstractmethod
    def remainder(self, K: float) -> float:
        pass
    
    def option_price(self, K: float, payoff: str='Call') -> float:
        """Returns the call/put price.
        """
        if payoff == "Call":
            return self.call_price(K)
        elif payoff == "Put":
            return self.call_price(K) + (K-self.f)
    
    def call_price(self, K: float) -> float:
        """Returns the call price obtained by averaging the BS call prices
        over the exponential distribution of vol transition time.
        """
        f = partial(self.integrand, K)
        return quad(f, 0.0, self.T_expiry)[0]+self.remainder(K)

class Latent_Markov_Vol(Vol_model):
    """Latent Markov volatility model.
    The process starts with volatility sigma_0 and randomly 
    jumps to sigma_1 with intensity lambda. The call pricing formula
    can be obtained by averaging standard pricing formulae over the
    exponential distribution of transition time from sigma_0 to sigma_1.
    """
    def __init__(self,
                 sigma_0: float=1.0,
                 sigmas: list=[], 
                 lambdas: list=[], 
                 f=None,
                 T_expiry: float=1.0,
                 vol_type=None,
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                ) -> None:
        
        super().__init__(f=f, 
                         T_expiry=T_expiry,
                         vol_type=vol_type,
                         logmoneyness_lo=logmoneyness_lo,
                         logmoneyness_hi=logmoneyness_hi,
                         K_lo=K_lo,
                         K_hi=K_hi,
                         strike_type=strike_type,
                         n_strikes=n_strikes
                        )
        self._sigma_0 = sigma_0
        self._sigmas = sigmas
        self._lambdas = lambdas
        self.check_size()
        
    def check_size(self):
        if len(self._sigmas) != len(self._lambdas):
            raise Exception('Not the same number of Volatilities and intensities!')
    
    @property
    def sigma_0(self) -> float:
        return self._sigma_0
    @sigma_0.setter
    def sigma_0(self, new_sigma_0: float) -> None:
        self._sigma_0 = new_sigma_0
        
    @property
    def sigmas(self) -> list:
        return self._sigmas
    @sigmas.setter
    def sigmas(self, new_sigmas: list) -> None:
        self._sigmas = new_sigmas
        self.check_size()
    
    @property
    def lambdas(self) -> list:
        return self._lambdas
    @lambdas.setter
    def lambdas(self, new_lambdas: list) -> None:
        self._lambdas = new_lambdas
        self.check_size()
        
    @property
    def intensity_sum(self) -> float:
        return np.sum(self.lambdas)
        
    @abc.abstractmethod
    def smile_func(self, K):
        pass
    
    def total_std(self, sigma_next: float, t: float) -> float:
        """Total standard deviation between 0 and T_expiry with a vol switch at t
        """
        return np.sqrt(sigma_next**2*self.T_expiry + (self.sigma_0**2-sigma_next**2)*t)
    
    @property
    def total_std_0(self) -> float:
        """Total standard deviation between 0 and T_expiry assuming no vol switch
        """
        return self.sigma_0*np.sqrt(self.T_expiry)
    
    @abc.abstractmethod
    def integrand(self, sigma_next: float, intensitynext: float, K: float, t: float) -> float:
        pass
    
    @abc.abstractmethod
    def remainder(self, intensitynext: float, K: float) -> float:
            pass
        
    def option_price(self, K, payoff='Call'):
        """Returns the call/put price.
        """
        if payoff == "Call":
            return self.call_price(K)
        elif payoff == "Put":
            return self.call_price(K) + (K-self.f)
    
    def call_price(self, K):
        """Returns the call price obtained by averaging the BS call prices
        over the exponential distribution of vol transition time.
        Given the volatility can only jump once, the call price is the average of the call price under BS_Markov_Simple for each couple (sigma_i, lambda_i), weighted by the probability that the jump to sigma_i is the first to occur. Markov chain property gives that the transition time are independent and follow exponential distributions, thus the probability that the jump to sigma_i occurs first is proportional to lambda_i.
        """
        prices = np.array(list(map(lambda param: quad(partial(self.integrand, param[0], param[1], K), 0.0, self.T_expiry)[0]+self.remainder(param[1], K), zip(self.sigmas, self.lambdas))))
        return np.dot(np.array(self.lambdas)/self.intensity_sum, prices)