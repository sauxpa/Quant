#!/usr/bin/env python
# coding: utf-8

import numpy as np
import abc
from volmodels import Vol_model

# SVI 
# Stochastic Volatility Inspired
# Not a diffusion model but rather a vol surface parametrization
class SVI_Raw(Vol_model):
    """Gatheral's raw SVI
    w(k) = a + b * ( rho * ( k - m ) + sqrt( ( k - m ) ^ 2 + sigma ^ 2 ) )
    where w(k) = sigma^2(k)*T_expiry is the total variance
    """
    def __init__(self, 
                 a: float=1.0, 
                 b: float=0.0,
                 rho: float=0.0, 
                 m: float=0.0,
                 sigma: float=0.0,
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
        super().__init__(
            f=f, 
            T_expiry=T_expiry, 
            vol_type=vol_type,                         
            logmoneyness_lo=logmoneyness_lo, 
            logmoneyness_hi=logmoneyness_hi,
            K_lo=K_lo, 
            K_hi=K_hi,
            strike_type=strike_type, 
            n_strikes=n_strikes
        )
        self._a = a
        self._b = b
        self._rho = rho
        self._m = m
        self._sigma = sigma
    
    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self, new_sigma):
        self._sigma = new_sigma
    
    @property
    def a(self):
        return self._a
    @a.setter
    def a(self, new_a):
        self._a = new_a
    
    @property
    def b(self):
        return self._b
    @b.setter
    def b(self, new_b):
        self._b = new_b

    @property
    def rho(self):
        return self._rho
    @rho.setter
    def rho(self, new_rho):
        self._rho = new_rho
        
    @property
    def m(self):
        return self._m
    @m.setter
    def m(self, new_m):
        self._m = new_m

    def __str__(self):
        return r'$a$={:.2f}, $b$={:.2f}, $\rho$={:.0%}, m={:.2f}, $\sigma$={:.2f}, f={:.2%}'        .format(self.a, self.b, self.rho, self.m, self.sigma, self.f)
    
    @abc.abstractmethod
    def smile_func(self, K):
        pass
    
class SVI_Raw_LN(SVI_Raw):
    """Gatheral's raw SVI in lognormal quoting
    """
    def __init__(self, 
                 a: float=1.0, 
                 b: float=0.0,
                 rho: float=0.0, 
                 m: float=0.0,
                 sigma: float=0.0,
                 f=None, 
                 T_expiry: float=1.0, 
                 logmoneyness_lo=None, 
                 logmoneyness_hi=None, 
                 K_lo=None, 
                 K_hi=None,
                 strike_type: str='logmoneyness', 
                 n_strikes: int=50,
                ) -> None:
        super().__init__(
            a=a,
            b=b,
            rho=rho,
            m=m,
            sigma=sigma,
            f=f, 
            T_expiry=T_expiry, 
            vol_type='LN',                         
            logmoneyness_lo=logmoneyness_lo, 
            logmoneyness_hi=logmoneyness_hi,
            K_lo=K_lo, 
            K_hi=K_hi,
            strike_type=strike_type, 
            n_strikes=n_strikes
        )
        
    @property
    def model_name(self):
        return 'SVI_Raw_LN'
    
    @property
    def ATM_LN(self):
        total_var = self.a + self.b * (self.rho * (-self.m) + np.sqrt((-self.m) ** 2 + self.sigma ** 2))
        return np.sqrt(total_var/self.T_expiry)
        
    @property
    def ATM(self):
        return self.ATM_LN
    
    def smile_func(self, K):
        """Implied vol comes from the SVI parametrization of the total variance
        """
        k = np.log(K/self.f)
        total_var = self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma ** 2))
        return np.sqrt(total_var/self.T_expiry)