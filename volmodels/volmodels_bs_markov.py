#!/usr/bin/env python
# coding: utf-8

import numpy as np
from volmodels import Latent_Markov_Vol_simple, Latent_Markov_Vol
from scipy.stats import norm

class BS_Markov_simple(Latent_Markov_Vol_simple):
    """Black-Scholes model with latent Markov volatility.
    The process starts with volatility sigma_0 and randomly 
    jumps to sigma_1 with intensity lambda. The call pricing formula
    can be obtained by averaging standard Black-Scholes prices over the
    exponential distribution of transition time from sigma_0 to sigma_1.
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
                         sigma_0=sigma_0,
                         sigma_1=sigma_1,
                         intensity=intensity,
                         vov=vov,
                         marking_mode=marking_mode,
                         T_expiry=T_expiry,
                         vol_type=vol_type,
                         logmoneyness_lo=logmoneyness_lo,
                         logmoneyness_hi=logmoneyness_hi,
                         K_lo=K_lo,
                         K_hi=K_hi,
                         strike_type=strike_type,
                         n_strikes=n_strikes
                        )
        
    def __str__(self) -> str:
        return r'$\sigma_0$={:.2%}, $\sigma_1$={:.2%}, $\lambda$={:.2%}, f={:.2%}'.format(self.sigma_0, self.sigma_1, self.intensity, self.f)
    
    def log_moneyness(self, K: float) -> float:
        return np.log(self.f/K)
    
    def d1(self, K: float, t: float) -> float:
        return 1/self.total_std(t)*self.log_moneyness(K)+0.5*self.total_std(t)
    
    def d2(self, K: float, t: float) -> float:
        return 1/self.total_std(t)*self.log_moneyness(K)-0.5*self.total_std(t)

    def d01(self, K: float) -> float:
        return 1/self.total_std_0*self.log_moneyness(K)+0.5*self.total_std_0
    def d02(self, K: float) -> float:
        return 1/self.total_std_0*self.log_moneyness(K)-0.5*self.total_std_0

    def integrand(self, K: float, t: float) -> float:
        return (self.f*norm.cdf(self.d1(K, t)) - K*norm.cdf(self.d2(K, t)))*self.intensity*np.exp(-self.intensity*t)

    def remainder(self, K: float) -> float:
        return (self.f * norm.cdf(self.d01(K)) - K * norm.cdf(self.d02(K))) * np.exp(-self.intensity*self.T_expiry)
    
class BS_Markov_simple_LN(BS_Markov_simple):
    """Black-Scholes with latent Markov volatility in lognormal quoting.
    """
    def __init__(self, 
                 sigma_0: float=1.0, 
                 sigma_1: float=1.0, 
                 intensity=None,
                 vov=None,
                 marking_mode='intensity',   
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
                             intensity=intensity,
                             vov=vov,
                             marking_mode=marking_mode,
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
        return 'BS_Markov_simple_LN'
    
    @property
    def ATM_LN(self) -> float:
        return self.smile_func(self.f)
        
    @property
    def ATM(self) -> float:
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
    
class BS_Markov_simple_N(BS_Markov_simple):
    """Black-Scholes with latent Markov volatility in normal quoting.
    """
    def __init__(self, 
                 sigma_0: float=1.0, 
                 sigma_1: float=1.0, 
                 intensity=None,
                 vov=None,
                 marking_mode='intensity',
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
                             intensity=intensity,
                             vov=vov,
                             marking_mode=marking_mode,
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
        return 'BS_Markov_simple_N'
    
    @property
    def ATM_LN(self) -> float:
        return self.smile_func(self.f)
        
    @property
    def ATM(self) -> float:
        return self.ATM_N
    
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
    
class BS_Markov(Latent_Markov_Vol):
    """Black-Scholes model with latent Markov volatility.
    The process starts with volatility sigma_0 and randomly 
    jumps to sigma_1 with intensity lambda. The call pricing formula
    can be obtained by averaging standard Black-Scholes prices over the
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
                         sigma_0=sigma_0,
                         sigmas=sigmas,
                         lambdas=lambdas,
                         T_expiry=T_expiry,
                         vol_type=vol_type,
                         logmoneyness_lo=logmoneyness_lo,
                         logmoneyness_hi=logmoneyness_hi,
                         K_lo=K_lo,
                         K_hi=K_hi,
                         strike_type=strike_type,
                         n_strikes=n_strikes
                        )
        
    def __str__(self):
        return r'$\sigma_0$={:.2%}'.format(self.sigma_0) \
    + r', $\sigma$=' + ', '.join(list(map(lambda x: '{:.2%}'.format(x), self.sigmas))) \
    + r', $\lambda$=' + ', '.join(list(map(lambda x: '{:.2f}'.format(x), self.lambdas))) \
    + ', f={:.2%}'.format(self.f)
    
    def log_moneyness(self, K: float) -> float:
        return np.log(self.f/K)
    
    def d1(self, sigma_next: float, K: float, t: float) -> float:
        return 1/self.total_std(sigma_next,t)*self.log_moneyness(K)+0.5*self.total_std(sigma_next,t)
    
    def d2(self, sigma_next: float, K: float, t: float) -> float:
        return 1/self.total_std(sigma_next,t)*self.log_moneyness(K)-0.5*self.total_std(sigma_next,t)
    
    def d01(self, K: float) -> float:
        return 1/self.total_std_0*self.log_moneyness(K)+0.5*self.total_std_0
    def d02(self, K: float) -> float:
        return 1/self.total_std_0*self.log_moneyness(K)-0.5*self.total_std_0

    def integrand(self, sigma_next: float, lambda_next: float, K: float, t: float) -> float:
        return (self.f * norm.cdf(self.d1(sigma_next, K, t)) - K * norm.cdf(self.d2(sigma_next, K, t)))*lambda_next*np.exp(-lambda_next*t)

    def remainder(self, lambda_next: float, K: float) -> float:
        return (self.f * norm.cdf(self.d01(K)) - K * norm.cdf(self.d02(K))) * np.exp(-lambda_next*self.T_expiry)
                          
class BS_Markov_LN(BS_Markov):
    """Black-Scholes with latent Markov volatility in lognormal quoting.
    """
    def __init__(self, 
                 sigma_0: float=1.0,
                 sigmas: list=[], 
                 lambdas: list=[], 
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
                             sigmas=sigmas,
                             lambdas=lambdas,
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
        return 'BS_Markov_LN'
    
    @property
    def ATM_LN(self) -> float:
        return self.smile_func(self.f)
        
    @property
    def ATM(self) -> float:
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
    
class BS_Markov_N(BS_Markov):
    """Black-Scholes with latent Markov volatility in lognormal quoting.
    """
    def __init__(self, 
                 sigma_0: float=1.0,
                 sigmas: list=[], 
                 lambdas: list=[], 
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
                             sigmas=sigmas,
                             lambdas=lambdas,
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
        return 'BS_Markov_N'
    
    @property
    def ATM_LN(self) -> float:
        return self.smile_func(self.f)
        
    @property
    def ATM(self) -> float:
        return self.ATM_N
    
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