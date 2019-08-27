#!/usr/bin/env python
# coding: utf-8

import numpy as np
import abc
from volmodels import Vol_model, ONE_BP

# Displaced lognormal
# Pure local volatility model here
class SLN_adjustable_backbone(Vol_model):
    """Shifted lognormal model with adjustable backbone as developped 
    in Andersen and Piterbarg in (16.5):
    dF_t = sigma*(shift*F_t+(mixing-shift)*F_0+(1-m)*L)*dW_t
    """
    def __init__(self, 
                 sigma: float=1.0,
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
                 n_strikes: int=50
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
        
        self._sigma = sigma
        self._shift = shift
        self._mixing = mixing
        self._L = L
    
    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self, new_sigma):
        self._sigma = new_sigma
    
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
        
    def __str__(self):
        return r'$\sigma$={:.2f}, shift={:.0%}, $\mixing$={:.0%}, L={:.2f}, f={:.2%}'        .format(self.sigma, self.shift, self.mixing, self.L, self.f)
    
    @abc.abstractmethod
    def smile_func(self, K):
        pass

class SLN_adjustable_backbone_LN(SLN_adjustable_backbone):
    """Shifted lognormal model with adjustable backbone in lognormal quoting
    """
    def __init__(self, 
                 sigma: float=1.0,
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
        super().__init__(sigma=sigma, 
                         shift=shift,
                         mixing=mixing,
                         L=L,
                         f=f, 
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
    def model_name(self):
        return 'SLN_adjustable_backbone_LN'
    
    @property
    def ATM_LN(self):
        local_vol_f = self.mixing*self.f+(1-self.mixing)*self.L
        return self.sigma*(self.mixing*self.f+(1-self.mixing)*self.L)/self.f    *(1+1/24*((-(self.shift/local_vol_f)**2+1/(self.f**2))*(self.sigma**2)*(local_vol_f**2))      *self.T_expiry)
        
    @property
    def ATM(self):
        return self.ATM_LN
    
    def smile_func(self, K):
        """Implied vol is the harmonic average of local vol on the path F_0 -> K
        """
        if np.abs(K-self.f)<ONE_BP/100:
            return self.ATM
        else:
            log_moneyness = np.log(self.f/K)
            f_avg = np.sqrt(self.f*K)
            int_inv_local_vol = np.log(                                       (self.mixing*self.f+(1-self.mixing)*self.L)/                                       (self.shift*K+(self.mixing-self.shift)*self.f                                        +(1-self.mixing)*self.L))
            local_vol_f_avg = self.shift*f_avg            +(self.mixing-self.shift)*self.f            +(1-self.mixing)*self.L
            gamma_1 = self.shift/local_vol_f_avg 
            
            return self.sigma*self.shift*log_moneyness/int_inv_local_vol        *(1+((1/(f_avg**2)-gamma_1**2)/24*(self.sigma**2)*(local_vol_f_avg**2))*self.T_expiry)