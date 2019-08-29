#!/usr/bin/env python
# coding: utf-8

import numpy as np
from volmodels import Implied_vol, Vol_model
from scipy.integrate import quad
from scipy.stats import norm
from functools import partial
import abc

class SLN_Markov_simple(Vol_model):
    """Shifted lognormal model with latent Markov volatility.
    The process starts with volatility sigma_0 and randomly 
    jumps to sigma_1 with intensity lambda. The call pricing formula
    can be obtained by averaging standard Black-Scholes prices over the
    exponential distribution of transition time from sigma_0 to sigma_1.
    dX_t = sigma*(shift*X_t+(mixing-shift)*X_0+(1-m)*L)*dW_t
    """
    def __init__(self,
                 sigma_0: float=1.0, 
                 sigma_1: float=1.0, 
                 lambda_: float=0.0, 
                 shift: float=0.0,
                 mixing: float=0.0,
                 L: float=0.0,
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
        self._lambda_ = lambda_
        self._shift = shift
        self._mixing = mixing
        self._L = L
    
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
    
    @property
    def shift(self) -> float:
        return self._shift
    @shift.setter
    def shift(self, new_shift: float) -> None:
        self._shift = new_shift
    
    @property
    def mixing(self) -> float:
        return self._mixing
    @mixing.setter
    def mixing(self, new_mixing: float) -> None:
        self._mixing = new_mixing
        
    @property
    def L(self) -> float:
        return self._L
    @L.setter
    def L(self, new_L: float) -> None:
        self._L = new_L
        
    @property
    def displaced_f(self) -> float:
        return self.mixing*self.f + (1-self.mixing)*self.L
        
    @property
    def displaced_sigma_0(self) -> float:
        return self.shift*self.sigma_0
     
    @property
    def displaced_sigma_1(self) -> float:
        return self.shift*self.sigma_1
    
    def __str__(self) -> str:
        return r'$\sigma_0$={:.2%}, $\sigma_1$={:.2%}, $\lambda$={:.2%}, shift={:.2f}, mixing={:.2f}, L={:.2f}, f={:.2%}'.format(self.sigma_0, self.sigma_1, self.lambda_, self.shift, self.mixing, self.L, self.f)
    
    @abc.abstractmethod
    def smile_func(self, K):
        pass
    
    def total_std(self, t: float) -> float:
        """Total standard deviation between 0 and T_expiry with a vol switch at t
        """
        return np.sqrt(self.displaced_sigma_1**2*self.T_expiry + (self.displaced_sigma_0**2-self.displaced_sigma_1**2)*t)
    
    @property
    def total_std_0(self) -> float:
        """Total standard deviation between 0 and T_expiry assuming no vol switch
        """
        return self.displaced_sigma_0*np.sqrt(self.T_expiry)
    
    def log_moneyness(self, K_shifted: float) -> float:
        return np.log(self.displaced_f/K_shifted)
    
    def d1(self, K_shifted: float, t: float) -> float:
        return 1/self.total_std(t)*self.log_moneyness(K_shifted)+0.5*self.total_std(t)
    
    def d2(self, K_shifted: float, t: float) -> float:
        return 1/self.total_std(t)*self.log_moneyness(K_shifted)-0.5*self.total_std(t)

    def d01(self, K_shifted: float) -> float:
        return 1/self.total_std_0*self.log_moneyness(K_shifted)+0.5*self.total_std_0
    def d02(self, K_shifted: float) -> float:
        return 1/self.total_std_0*self.log_moneyness(K_shifted)-0.5*self.total_std_0
    
    def integrand(self, K_shifted: float, t: float) -> float:
        return (self.displaced_f * norm.cdf(self.d1(K_shifted, t)) - K_shifted * norm.cdf(self.d2(K_shifted, t)))*self.lambda_*np.exp(-self.lambda_*t)

    def remainder(self, K_shifted: float) -> float:
        return (self.displaced_f * norm.cdf(self.d01(K_shifted)) - K_shifted * norm.cdf(self.d02(K_shifted))) * np.exp(-self.lambda_*self.T_expiry)
    
    def option_price(self, K, payoff='Call'):
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
        K_shifted = self.shift*K + (self.mixing-self.shift)*self.f+(1-self.mixing)*self.L
        f = partial(self.integrand, K_shifted)
        return (quad(f, 0.0, self.T_expiry)[0]+self.remainder(K_shifted))/self.shift
    
class SLN_Markov_simple_LN(SLN_Markov_simple):
    """Shifted lognormal with latent Markov volatility in lognormal quoting.
    """
    def __init__(self, 
                 sigma_0: float=1.0, 
                 sigma_1: float=1.0, 
                 lambda_: float=0.0, 
                 shift: float=0.0,
                 mixing: float=0.0,
                 L: float=0.0,
                 f=None,
                 T_expiry: float=1.0,
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                ) -> None:
            
            super().__init__(f=f,
                             sigma_0=sigma_0,
                             sigma_1=sigma_1,
                             lambda_=lambda_,
                             shift=shift,
                             mixing=mixing,
                             L=L,
                             T_expiry=T_expiry,
                             vol_type='LN',
                             logmoneyness_lo=logmoneyness_lo,
                             logmoneyness_hi=logmoneyness_hi,
                             K_lo=K_lo,
                             K_hi=K_hi,
                             strike_type=strike_type,
                             n_strikes=n_strikes,
                        )
    
    @property
    def model_name(self) -> str:
        return 'SLN_Markov_simple_LN'
    
    @property
    def ATM_LN(self) -> float:
        return self.smile_func(self.f)
        
    @property
    def ATM(self) -> float:
        return self.ATM_LN

    def smile_func(self, K: float) -> float:
        """Implied vol smile converted from option prices
        K: strike
        """
        if K < self.f:
            payoff = 'Put'
        else:
            payoff = 'Call'
            
        price = self.option_price(K, payoff=payoff)
        return self.IV.vol_from_price(price, self.f, K, self.T_expiry, payoff=payoff)
    
class SLN_Markov_simple_N(SLN_Markov_simple):
    """Shifted lognormal with latent Markov volatility in normal quoting.
    """
    def __init__(self, 
                 sigma_0: float=1.0, 
                 sigma_1: float=1.0, 
                 lambda_: float=0.0, 
                 shift: float=0.0,
                 mixing: float=0.0,
                 L: float=0.0,
                 f=None,
                 T_expiry: float=1.0,
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                ) -> None:
            
            super().__init__(f=f,
                             sigma_0=sigma_0,
                             sigma_1=sigma_1,
                             lambda_=lambda_,
                             shift=shift,
                             mixing=mixing,
                             L=L,
                             T_expiry=T_expiry,
                             vol_type='N',
                             logmoneyness_lo=logmoneyness_lo,
                             logmoneyness_hi=logmoneyness_hi,
                             K_lo=K_lo,
                             K_hi=K_hi,
                             strike_type=strike_type,
                             n_strikes=n_strikes,
                        )
    
    @property
    def model_name(self) -> str:
        return 'SLN_Markov_simple_N'
    
    @property
    def ATM_N(self) -> float:
        return self.smile_func(self.f)
        
    @property
    def ATM(self) -> float:
        return self.ATM_N

    def smile_func(self, K: float) -> float:
        """Implied vol smile converted from option prices
        K: strike
        """
        if K < self.f:
            payoff = 'Put'
        else:
            payoff = 'Call'
            
        price = self.option_price(K, payoff=payoff)
        return self.IV.vol_from_price(price, self.f, K, self.T_expiry, payoff=payoff)    