#!/usr/bin/env python
# coding: utf-8

import numpy as np
from volmodels import Implied_vol, Vol_model
from scipy.integrate import quad
from scipy.special import erf
from functools import partial
import abc

ONE_BP = 1e-4
ONE_PCT = 1e-2

class BS_Markov_simple(Vol_model):
    """Black-Scholes model with latent Markov volatility.
    The process starts with volatility sigma_0 and randomly 
    jumps to sigma_1 with intensity lambda. The call pricing formula
    can be obtained by averaging standard Black-Scholes prices over the
    exponential distribution of transition time from sigma_0 to sigma_1.
    """
    def __init__(self,
                 sigma_0: float=1.0, 
                 sigma_1: float=1.0, 
                 lambda_: float=0.0, 
                 f=None,
                 T_expiry: float=1.0,
                 vol_type=None,
                 moneyness_lo=None,
                 moneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                ) -> None:
        
        super().__init__(f=f, 
                         T_expiry=T_expiry,
                         vol_type=vol_type,
                         moneyness_lo=moneyness_lo,
                         moneyness_hi=moneyness_hi,
                         K_lo=K_lo,
                         K_hi=K_hi,
                         strike_type=strike_type,
                         n_strikes=n_strikes
                        )
        self._sigma_0 = sigma_0
        self._sigma_1 = sigma_1
        self._lambda_ = lambda_
    
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
    def lambda_(self) -> float:
        return self._lambda_
    @lambda_.setter
    def lambda_(self, new_lambda_: float) -> None:
        self._lambda_ = new_lambda_
        
    def __str__(self):
        return r'$\sigma_0$={:.2%}, $\sigma_1$={:.2%}, $\lambda$={:.2%}, f={:.2%}'.format(self.sigma_0, self.sigma_1, self.lambda_, self.f)
    
    @abc.abstractmethod
    def smile_func(self, K):
        pass
    
    def N(self, x: float) -> float:
        return 0.5*(1+erf(x/np.sqrt(2)))
    
    def log_moneyness(self, K: float) -> float:
        return np.log(self.f/K)
    
    def d1(self, K: float, t: float) -> float:
        total_std = np.sqrt(self.sigma_1**2*self.T_expiry + (self.sigma_0**2-self.sigma_1**2)*t)
        return 1/total_std*self.log_moneyness(K)+0.5*total_std
    
    def d2(self, K: float, t: float) -> float:
        total_std = np.sqrt(self.sigma_1**2*self.T_expiry + (self.sigma_0**2-self.sigma_1**2)*t)
        return 1/total_std*self.log_moneyness(K)-0.5*total_std

    @property
    def total_std_0(self) -> float:
        return self.sigma_0*np.sqrt(self.T_expiry)
    
    def d01(self, K: float) -> float:
        return 1/self.total_std_0*self.log_moneyness(K)+0.5*self.total_std_0
    def d02(self, K: float) -> float:
        return 1/self.total_std_0*self.log_moneyness(K)-0.5*self.total_std_0

    def integrand(self, K: float, t: float) -> float:
        return (self.f * self.N(self.d1(K, t)) - K * self.N(self.d2(K, t)))*self.lambda_*np.exp(-self.lambda_*t)

    def remainder(self, K: float) -> float:
        return (self.f * self.N(self.d01(K)) - K * self.N(self.d02(K))) * np.exp(-self.lambda_*self.T_expiry)
    
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
        """
        f = partial(self.integrand, K)
        return quad(f, 0.0, self.T_expiry)[0]+self.remainder(K)
    
class BS_Markov_simple_LN(BS_Markov_simple):
    """Black-Scholes with latent Markov volatility in lognormal quoting.
    """
    def __init__(self, 
                 sigma_0: float=1.0, 
                 sigma_1: float=1.0, 
                 lambda_: float=0.0, 
                 f=None,
                 T_expiry: float=1.0,
                 moneyness_lo=None,
                 moneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                ) -> None:
            
            super().__init__(f=f,
                             sigma_0=sigma_0,
                             sigma_1=sigma_1,
                             lambda_=lambda_,
                             T_expiry=T_expiry,
                             vol_type='LN',
                             moneyness_lo=moneyness_lo,
                             moneyness_hi=moneyness_hi,
                             K_lo=K_lo,
                             K_hi=K_hi,
                             strike_type=strike_type,
                             n_strikes=n_strikes,
                        )
    
    @property
    def model_name(self):
        return 'BS_Markov_LN'
    
    @property
    def ATM_LN(self):
        return self.smile_func(self.f)
        
    @property
    def ATM(self):
        return self.ATM_LN
    
    def smile_func(self, K):
        """Implied vol smile converted from option prices
        K: strike
        """
        if K < self.f:
            payoff = 'Put'
        else:
            payoff = 'Call'
            
        price = self.option_price(K, payoff=payoff)
        return self.IV.vol_from_price(price, self.f, K, self.T_expiry, payoff=payoff)